import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np


def initialize_weights(m: nn.Module) -> None:
    """
    Initializes the weights of the given module.

    Args:
    - m (nn.Module): the module to initialize weights of

    Returns:
    - None
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0, 0.1)
        m.bias.data.fill_(0.00)


class FBSDESolver:
    def __init__(
            self,
            pde,
            layer_sizes,
            learning_rate,
            solving_method
    ):
        """
        Initializes the FBSDESolver.

        Args:
        - pde : the partial differential equation to solve
        - layer_sizes (list[int]): list of sizes of hidden layers
        - learning_rate (float): learning rate for optimization
        - solving_method (str): method to solve the PDE ('Monte-Carlo', 'Deep-Time-SetTransformer', 'Basic')
        """
        self.pde = pde
        self.layer_size = layer_sizes
        self.learning_rate = learning_rate
        self.method = solving_method
        if solving_method == 'Monte-Carlo':
            self.solver = TimeDependentNetworkMonteCarlo(pde.dim, self.layer_size, pde.dim, pde.sigma)
        elif solving_method == 'Deep-Time-SetTransformer':
            self.solver = DeepTimeSetTransformer(1)
        elif solving_method == 'Basic':
            self.solver = TimeDependentNetwork(pde.dim, self.layer_size, pde.dim)

        self.optimizer = torch.optim.Adam(self.solver.parameters(), lr=self.learning_rate)

    def solve(
            self,
            num_iterations,
            batch_size,
            init,
            device,
            sample_size=None
    ):
        """
        Solves the PDE.

        Args:
        - num_iterations (int): number of iterations for optimization
        - batch_size (int): batch size for training
        - init (torch.Tensor): initial value
        - device (torch.device): device to perform calculations on ('cpu', 'cuda')
        - sample_size (int, optional): sample size for Monte-Carlo method

        Returns:
        - list[torch.Tensor]: list of losses during optimization
        - list[torch.Tensor]: list of initial values during optimization
        """
        losses = []
        inits = []
        self.solver.to(device)

        torch.seed()
        self.solver.train()

        for j in range(num_iterations):
            print(j + 1)

            dw, y = self.pde.sample(batch_size)
            y = torch.tensor(y)
            y = y.to(torch.float32)
            dw = torch.tensor(dw).to(torch.float32)
            dw = dw.to(device)
            t = torch.ones((batch_size, 1)).to(device)

            self.optimizer.zero_grad()

            y = y.to(device)

            y_terminal = torch.ones((batch_size,)).to(device) * init.to(device)

            y_terminal = y_terminal.to(device)
            coef = torch.ones((batch_size,)).to(device)
            dw_coef = torch.zeros((batch_size,)).to(device)
            L1 = 0.0
            if self.method == 'Monte-Carlo':
                num_sample = 5000
                primary_sample = torch.unsqueeze(torch.randn(size=[num_sample, self.pde.dim]), dim=0).to(device)
            for z in range(self.pde.num_time_interval):
                S0 = y[:, :, z]

                t0 = t * self.pde.delta_t * z
                if self.method == 'Monte-Carlo':
                    ders = []
                    time_length = (self.pde.num_time_interval - z) * self.pde.delta_t
                    current_state = torch.unsqueeze(S0, dim=1).to(device)
                    dw_sample = primary_sample * torch.tensor(np.sqrt(time_length)).to(device)
                    k = (1 + self.pde.mu_bar * time_length) * current_state + (self.pde.sigma * current_state * dw_sample)
                    ddw = dw_sample
                    means = torch.mean((self.pde.terminal_for_sample(k)), dim=1, keepdim=True)
                    mins = torch.mean((self.pde.terminal_for_sample(k) - means) * ddw, dim=1) / torch.tensor(
                        (self.pde.num_time_interval - z) * self.pde.delta_t).to(device)
                    ders = torch.tensor(mins).to(torch.float32).to(device)
                    out1 = self.solver(t0, S0, ders)
                elif self.method == 'Deep-Time-SetTransformer':
                    t = torch.ones((batch_size, 1, 1)).to(device)
                    t0 = t * self.pde.delta_t * z
                    S0 = S0.reshape(S0.size()[0], 100, 1).requires_grad_(True)
                    out1 = self.solver(t0, S0)
                    out1 = torch.squeeze(autograd.grad(out1.sum(), S0, create_graph=True)[0])

                    S0 = torch.squeeze(S0)

                    out1 = out1 * S0 * self.pde.sigma
                elif self.method == 'Basic':
                    out1 = self.solver(t0, S0)
                else:
                    print('Model type is not true')

                samp = dw[:, :, z].to(device)

                nsim = S0.size()[0]
                interest_rate = torch.squeeze(self.pde.r_u(t0, S0, y_terminal, out1).to(device))
                hz = self.pde.h_z(t0, S0, y_terminal, out1)
                if z > 0:
                    dw_coef = dw_coef * (1 + interest_rate * self.pde.delta_t) + torch.squeeze(
                        torch.bmm(torch.unsqueeze(out1, 1), torch.unsqueeze(samp, 2))) + torch.squeeze(
                        self.pde.h_z(t0, S0, y_terminal, out1)).to(device) * self.pde.delta_t
                else:
                    dw_coef = torch.squeeze(
                        torch.bmm(torch.unsqueeze(out1, 1), torch.unsqueeze(samp, 2))) + torch.squeeze(
                        self.pde.h_z(t0, S0, y_terminal, out1)).to(device) * self.pde.delta_t
                coef = coef * (1 + interest_rate * self.pde.delta_t)
                y_terminal = y_terminal * (1 + interest_rate * self.pde.delta_t) + torch.squeeze(
                    self.pde.h_z(t0, S0, y_terminal, out1).to(device)) * self.pde.delta_t + torch.squeeze(
                    torch.bmm(torch.unsqueeze(out1, 1), torch.unsqueeze(samp, 2)))

            S0 = y[:, :, self.pde.num_time_interval]
            t0 = t * self.pde.delta_t * self.pde.num_time_interval

            payoff = self.pde.terminal(t0, S0)

            coef = coef.to(device)
            inits.append(init)
            init = (torch.mean((torch.squeeze(payoff) - dw_coef.detach()) / coef).detach())
            print(init)

            loss = torch.mean(torch.square(torch.squeeze(payoff) - torch.squeeze(y_terminal)))

            losses.append(loss.detach())

            print(loss)

            loss.backward()

            self.optimizer.step()

        return losses, inits
