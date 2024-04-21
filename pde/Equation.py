import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
from torch.nn import Module, Linear, BatchNorm1d, Tanh
from numba import cuda
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import math
"""
In this section we have used code available online in : https://github.com/frankhan91/DeepBSDE

"""

class Equation:
    """
    Base class for defining PDE related function.

    Args:
    eqn_config (dict): dictionary containing PDE configuration parameters

    Attributes:
    dim (int): dimensionality of the problem
    total_time (float): total time horizon
    num_time_interval (int): number of time steps
    delta_t (float): time step size
    sqrt_delta_t (float): square root of time step size
    y_init (None): initial value of the function
    """

    def __init__(self, eqn_config: dict):
        self.dim = eqn_config['dim']
        self.total_time = eqn_config['total_time']
        self.num_time_interval = eqn_config['num_time_interval']
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None

    def sample(self, num_sample: int) -> Tensor:
        """
        Sample forward SDE.

        Args:
        num_sample (int): number of samples to generate

        Returns:
        Tensor: tensor of size [num_sample, dim+1] containing samples
        """
        raise NotImplementedError

    def r_u(self, t: float, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        """
        Interest rate in the PDE.

        Args:
        t (float): current time
        x (Tensor): tensor of size [batch_size, dim] containing space coordinates
        y (Tensor): tensor of size [batch_size, 1] containing function values
        z (Tensor): tensor of size [batch_size, dim] containing gradients

        Returns:
        Tensor: tensor of size [batch_size, 1] containing generator values
        """
        raise NotImplementedError

    def h_z(self, t,x,y,z: Tensor) -> Tensor:
        """
        Function to compute H(z) in the PDE.

        Args:
        h (float): value of H function
        z (Tensor): tensor of size [batch_size, dim] containing gradients

        Returns:
        Tensor: tensor of size [batch_size, dim] containing H(z)
        """
        raise NotImplementedError

    def terminal(self, t: float, x: Tensor) -> Tensor:
        """
        Terminal condition of the PDE.

        Args:
        t (float): current time
        x (Tensor): tensor of size [batch_size, dim] containing space coordinates

        Returns:
        Tensor: tensor of size [batch_size, 1] containing terminal values
        """
        raise NotImplementedError


class PricingDefaultRisk(Equation):
  """
  Args:
  eqn_config (dict): dictionary containing PDE configuration parameters
  """
  def __init__(self, eqn_config):
    super(PricingDefaultRisk, self).__init__(eqn_config)
    self.x_init = np.ones(self.dim) * 100.0
    self.sigma = 0.2
    self.rate = 0.02   # interest rate R
    self.delta = 2.0 / 3
    self.gammah = 0.2
    self.gammal = 0.02
    self.mu_bar = 0.02
    self.K = 100
    self.vh = 50.0
    self.vl = 70.0
    self.slope = (self.gammah - self.gammal) / (self.vh - self.vl)

  def sample(self, num_sample)-> tuple:
    """
    Sample forward SDE.

    Args:
    num_sample (int): number of samples to generate

    Returns:
    tuple: tuple of two tensors: dw_sample of size [num_sample, dim, num_time_interval] and
    x_sample of size [num_sample, dim, num_time_interval+1]
    """

    dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
    x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
    x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
    for i in range(self.num_time_interval):
        x_sample[:, :, i + 1] = (1 + self.mu_bar * self.delta_t) * x_sample[:, :, i] + (
            self.sigma_matrix(x_sample[:, :, i]) * dw_sample[:, :, i])
    return dw_sample, x_sample

  def r_u(self, t, x, y, z)-> torch.Tensor:
    """
    Interest rate in the PDE.

    Args:
    t (float): current time
    x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates
    y (torch.Tensor): tensor of size [batch_size, 1] containing function values
    z (torch.Tensor): tensor of size [batch_size, dim] containing gradients

    Returns:
    torch.Tensor: tensor of size [batch_size, 1] containing generator values
    """
    piecewise_linear = nn.ReLU()(
        nn.ReLU()(y - self.vh) * self.slope + self.gammah - self.gammal) + self.gammal
    return ((1 - self.delta) * piecewise_linear  + self.rate)

  def h_z(self,t,x,y,z)-> torch.Tensor:
      """
      Function to compute $h^T Z$ in the PDE.

      Args:
      t (float): current time
      x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates
      y (torch.Tensor): tensor of size [batch_size, 1] containing function value
      z (torch.Tensor): tensor of size [batch_size, dim] containing gradients

      Returns:
      torch.Tensor: tensor of size [batch_size, 1] containing H(z)
      """
      return torch.zeros((x.size()[0],1))

  def sigma_matrix(self,x):
    return self.sigma * x


  def terminal(self, t, x)-> torch.Tensor:
    """
    Terminal condition of the PDE.

    Args:
    t (float): current time
    x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates

    Returns:
    torch.Tensor: tensor of size [batch_size, 1] containing terminal values
    """
    return nn.ReLU()(torch.min(x, 1)[0])
  def terminal_for_sample(self, x)-> torch.Tensor:
    """
    Terminal condition of the PDE.

    Args:
    x (torch.Tensor): tensor of size [num_sample,batch_size, dim] containing space coordinates

    Returns:
    torch.Tensor: tensor of size [num_sample ,batch_size, 1] containing terminal values
    """
    return nn.ReLU()(torch.min(x, 2 , keepdim= True)[0])


class HJBLQ(Equation):
  """

  Args:
  eqn_config (dict): dictionary containing PDE configuration parameters
  """

  def __init__(self, eqn_config: dict):
    super().__init__(eqn_config)

    self.x_init = np.zeros(self.dim)
    self.sigma = np.sqrt(2.0)
    self.lambd = 1.0

  def sample(self, num_sample: int) -> tuple:
    """
    Sample forward SDE.

    Args:
    num_sample (int): number of samples to generate

    Returns:
    tuple: tuple of two tensors: dw_sample of size [num_sample, dim, num_time_interval] and
    x_sample of size [num_sample, dim, num_time_interval+1]
    """
    dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
    x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
    x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
    for i in range(self.num_time_interval):
        x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
    return dw_sample, x_sample

  def r_u(self, t: float, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Interest rate in the PDE.

    Args:
    t (float): current time
    x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates
    y (torch.Tensor): tensor of size [batch_size, 1] containing function values
    z (torch.Tensor): tensor of size [batch_size, dim] containing gradients

    Returns:
    torch.Tensor: tensor of size [batch_size, 1] containing generator values
    """
    return torch.ones((x.size()[0],)) * 0

  def h_z(self, t: float, x: torch.Tensor,y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Function to compute <h,z> in the PDE.

    Args:
    t (float): current time
    x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates
    y (torch.Tensor): tensor of size [batch_size, 1] containing function value
    z (torch.Tensor): tensor of size [batch_size, dim] containing gradients

    Returns:
    torch.Tensor: tensor of size [batch_size, 1] containing H(z)
    """
    return torch.sum(torch.square(z), dim=1)/ (self.sigma**2)

  def terminal(self, t: float, x: torch.Tensor) -> torch.Tensor:
    """
    Terminal condition of the PDE.

    Args:
    t (float): current time
    x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates

    Returns:
    torch.Tensor: tensor of size [batch_size, 1] containing terminal values
    """
    return torch.log(0.5 * (1 + torch.norm(x, dim=1)**2))
  def sigma_matrix(self,x):
    return self.sigma
def terminal_for_sample(self, x)-> torch.Tensor:
    """
    Terminal condition of the PDE.

    Args:
    x (torch.Tensor): tensor of size [num_sample,batch_size, dim] containing space coordinates

    Returns:
    torch.Tensor: tensor of size [num_sample ,batch_size, 1] containing terminal values
    """
    return torch.sum(x ** 2, dim=2 ,keepdim=True )

class BlackScholesBarenblatt(Equation):
  """

  Args:
  eqn_config (dict): dictionary containing PDE configuration parameters
  """

  def __init__(self, eqn_config: dict):
    super().__init__(eqn_config)
    self.x_init = np.ones(self.dim) * np.array([1.0 / (1.0 + i % 2) for i in range(self.dim)])
    self.sigma = 0.4
    self.rate = 0.05   # interest rate R
    self.mu_bar = 0.0

  def sample(self, num_sample: int) -> tuple:
    """
    Sample forward SDE.

    Args:
    num_sample (int): number of samples to generate

    Returns:
    tuple: tuple of two tensors: dw_sample of size [num_sample, dim, num_time_interval] and
    x_sample of size [num_sample, dim, num_time_interval+1]
    """
    dw_sample = np.random.normal(size=(num_sample, self.dim, self.num_time_interval)) * self.sqrt_delta_t
    x_sample = np.zeros((num_sample, self.dim, self.num_time_interval + 1))
    x_sample[:, :, 0] = np.ones((num_sample, self.dim)) * self.x_init
    for i in range(self.num_time_interval):
        x_sample[:, :, i + 1] = (1 + self.mu_bar * self.delta_t) * x_sample[:, :, i] + (
            self.sigma_matrix(x_sample[:, :, i]) * dw_sample[:, :, i])
    return dw_sample, x_sample

  def r_u(self, t: float, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Interest rate in the PDE.

    Args:
    t (float): current time
    x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates
    y (torch.Tensor): tensor of size [batch_size, 1] containing function values
    z (torch.Tensor): tensor of size [batch_size, dim] containing gradients

    Returns:
    torch.Tensor: tensor of size [batch_size, 1] containing generator values
    """
    return torch.ones((x.size()[0],)) * self.rate

  def h_z(self, t: float, x: torch.Tensor,y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Function to compute <h,z> in the PDE.

    Args:
    t (float): current time
    x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates
    y (torch.Tensor): tensor of size [batch_size, 1] containing function value
    z (torch.Tensor): tensor of size [batch_size, dim] containing gradients

    Returns:
    torch.Tensor: tensor of size [batch_size, 1] containing H(z)
    """
    return -1 * torch.sum(z, dim=1) * self.rate / self.sigma

  def terminal(self, t: float, x: torch.Tensor) -> torch.Tensor:
    """
    Terminal condition of the PDE.

    Args:
    t (float): current time
    x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates

    Returns:
    torch.Tensor: tensor of size [batch_size, 1] containing terminal values
    """
    return torch.sum(x ** 2, dim=1)
  def sigma_matrix(self,x):
    return self.sigma * x
  def terminal_for_sample(self, x)-> torch.Tensor:
    """
    Terminal condition of the PDE.

    Args:
    x (torch.Tensor): tensor of size [num_sample,batch_size, dim] containing space coordinates

    Returns:
    torch.Tensor: tensor of size [num_sample ,batch_size, 1] containing terminal values
    """
    return torch.sum(x ** 2, dim=2 ,keepdim=True )

class PricingDiffRate(Equation):
    """
    Nonlinear Black-Scholes equation with different interest rates for borrowing and lending
    in Section 4.4 of Comm. Math. Stat. paper doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(PricingDiffRate, self).__init__(eqn_config)
        self.x_init = np.ones(self.dim) * 100
        self.sigma = 0.2
        self.mu_bar = 0.06
        self.rl = 0.04
        self.rb = 0.06
        self.alpha = 1.0 / self.dim

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        factor = np.exp((self.mu_bar-(self.sigma**2)/2)*self.delta_t)
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        return dw_sample, x_sample

    def r_u(self, t: float, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
      temp = torch.sum(z, dim= 1) / self.sigma - y
      return torch.where(temp > 0 , self.rb , self.rl)
    def h_z(self, t: float, x: torch.Tensor,y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
      """
      Function to compute <h,z> in the PDE.

      Args:
      t (float): current time
      x (torch.Tensor): tensor of size [batch_size, dim] containing space coordinates
      y (torch.Tensor): tensor of size [batch_size, 1] containing function value
      z (torch.Tensor): tensor of size [batch_size, dim] containing gradients

      Returns:
      torch.Tensor: tensor of size [batch_size, 1] containing H(z)
      """
      temp = torch.sum(z, dim= 1) / self.sigma - y
      return torch.where(temp > 0 , (self.mu_bar - self.rb) * torch.sum(z, dim=1) / self.sigma ,(self.mu_bar - self.rl) * torch.sum(z, dim=1) / self.sigma)


    def terminal(self, t, x):
        temp = torch.max(x, 1)[0]
        return nn.ReLU()(temp - 120) - 2*nn.ReLU()(temp - 150)
    def sigma_matrix(self,x):
      return self.sigma * x
    def terminal_for_sample(self, x)-> torch.Tensor:
        """
        Terminal condition of the PDE.

        Args:
        x (torch.Tensor): tensor of size [num_sample,batch_size, dim] containing space coordinates

        Returns:
        torch.Tensor: tensor of size [num_sample ,batch_size, 1] containing terminal values
        """
        temp = torch.max(x, 2 , keep_dim = True)[0]
        return torch.maximum(temp - 120, torch.tensor(0))
