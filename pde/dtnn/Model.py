import torch

from torch.nn import Module, Linear, BatchNorm1d, Tanh
class TimeNet(Module):
    """
    Neural network model for time dimension
    """
    def __init__(
            self,
            output_dim: int
    ):
        """
        Initialize the neural network model with layers

        :param output_dim: The output dimension of the neural network
        :type output_dim: int
        """
        super(TimeNet, self).__init__()
        self.layer1 = Linear(4, 100)
        self.layer2 = Linear(100, 150)
        self.layer3 = Linear(150, 200)
        self.layer4 = Linear(200, 300)
        self.layer5 = Linear(300, 200)
        self.layer6 = Linear(200, output_dim)
        self.tanh1 = Tanh()
        self.tanh2 = Tanh()
        self.tanh3 = Tanh()
        self.tanh4 = Tanh()
        self.tanh5 = Tanh()
        self.tanh6 = Tanh()

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward propagation through the network.

        :param x: Input tensor
        :type x: torch.Tensor
        :return: Output tensor
        :rtype: torch.Tensor
        """
        x = self.layer1(x)
        x = self.tanh1(x)
        x = self.layer2(x)
        x = self.tanh2(x)
        x = self.layer3(x)
        x = self.tanh3(x)
        x = self.layer4(x)
        x = self.tanh4(x)
        x = self.layer5(x)
        x = self.tanh5(x)
        x = self.layer6(x)
        x = self.tanh6(x)
        return x


class Net1(Module):
    """
    A class for defining a neural network with a single linear layer.
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int
    ):
        """
        Initialize the network with a single linear layer.

        :param input_dim: Number of input features
        :type input_dim: int
        :param output_dim: Number of output features
        :type output_dim: int
        """
        super(Net1, self).__init__()
        self.layer = Linear(input_dim, output_dim)
        self.bn = BatchNorm1d(output_dim)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward propagation through the network.

        :param x: Input tensor of shape (batch_size, input_dim)
        :type x: torch.Tensor
        :return: Output tensor of shape (batch_size, output_dim)
        :rtype: torch.Tensor
        """
        x = self.layer(x)
        return x
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
For implementing Deep-Time-SetTransformer we have used code written by authors of Set-Transformer available at : https://github.com/juho-lee/set_transformer
"""

class MAB(nn.Module):
    def __init__(
        self,
        dim_q: int,
        dim_k: int,
        dim_v: int,
        num_heads: int,
        ln: bool = False
    ):
        """
        Multi-Head Self Attention Block.

        :param dim_q: Dimension of query
        :param dim_k: Dimension of key
        :param dim_v: Dimension of value
        :param num_heads: Number of attention heads
        :param ln: Whether to use Layer Normalization
        """
        super(MAB, self).__init__()
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_k, dim_v)
        if ln:
            self.ln0 = nn.LayerNorm(dim_v)
            self.ln1 = nn.LayerNorm(dim_v)
        self.fc_o = nn.Linear(dim_v, dim_v)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward propagation.

        :param q: Query tensor
        :param k: Key tensor
        :return: Output tensor
        """
        q = self.fc_q(q)
        k, v = self.fc_k(k), self.fc_v(k)

        dim_split = self.dim_v // self.num_heads
        q_ = torch.cat(q.split(dim_split, 2), 0)
        k_ = torch.cat(k.split(dim_split, 2), 0)
        v_ = torch.cat(v.split(dim_split, 2), 0)

        a = torch.softmax(q_.bmm(k_.transpose(1,2))/math.sqrt(self.dim_v), 2)
        o = torch.cat((q_ + a.bmm(v_)).split(q.size(0), 0), 2)
        o = o if getattr(self, 'ln0', None) is None else self.ln0(o)
        o = o + F.relu(self.fc_o(o))
        o = o if getattr(self, 'ln1', None) is None else self.ln1(o)
        return o

class SAB(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        ln: bool = False
    ):
        """
        Self Attention Block.

        :param dim_in: Input dimension
        :param dim_out: Output dimension
        :param num_heads: Number of attention heads
        :param ln: Whether to use Layer Normalization
        """
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward propagation.

        :param x: Input tensor
        :return: Output tensor
        """
        return self.mab(x, x)

class ISAB(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        num_inds: int,
        ln: bool = False
    ):
        """
        Induced Self Attention Block.

        :param dim_in: Input dimension
        :param dim_out: Output dimension
        :param num_heads: Number of attention heads
        :param num_inds: Number of inducing points
        :param ln: Whether to use Layer Normalization
        """
        super(ISAB, self).__init__()
        self.i = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.i)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward propagation.

        :param x: Input tensor
        :return: Output tensor
        """
        h = self.mab0(self.i.repeat(x.size(0), 1, 1), x)
        return self.mab1(x, h)

class PMA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_seeds: int,
        ln: bool = False
    ):
        """
        Pooling Multihead Attention.

        :param dim: Dimension of input and output
        :param num_heads: Number of attention heads
        :param num_seeds: Number of seed vectors
        :param ln: Whether to use Layer Normalization
        """
        super(PMA, self).__init__()
        self.s = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.s)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward propagation.

        :param x: Input tensor
        :return: Output tensor
        """
        return self.mab(self.s.repeat(x.size(0), 1, 1), x)
import torch
import torch.nn as nn


class TimeNetForSet(nn.Module):
    """
    Neural network model for time dimension.

    Args:
        in_features (int): The input features dimension. Default is 1.
        out_features (int): The output features dimension. Default is 64.
    """
    def __init__(self,
                 in_features: int = 1,
                 out_features: int = 64):
        super(TimeNetForSet, self).__init__()
        self.feature = nn.Linear(in_features=in_features, out_features=out_features)
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        self.layer4 = nn.Linear(10, out_features)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self,
                t: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            t (torch.Tensor): Input tensor for time dimension.
            x (torch.Tensor): Input tensor for features.

        Returns:
            torch.Tensor: Output tensor.
        """
        t = self.relu1(self.layer1(t))
        t = self.relu2(self.layer2(t))
        t = self.relu3(self.layer3(t))
        t = self.layer4(t)
        x = self.feature(x)

        x = (torch.exp(t)) * x

        return x

    def freeze(self):
        """Freezes the feature parameters."""
        for param in self.feature.parameters():
            param.requires_grad = False


class DeepTimeSetTransformer(nn.Module):
    def __init__(self, input_dim: int):
        super(DeepTimeSetTransformer, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.time_layer1 = TimeNetForSet(in_features=input_dim, out_features=32)
        self.time_layer2 = TimeNetForSet(in_features=32, out_features=32)
        self.time_layer3 = TimeNetForSet(in_features=32, out_features=32)
        self.time_layer4 = TimeNetForSet(in_features=32, out_features=32)
        self.time_layer5 = TimeNetForSet(in_features=32, out_features=1)

        self.layer1 = nn.Linear(in_features=input_dim, out_features=32)
        self.layer2 = nn.Linear(in_features=32, out_features=32)
        self.layer3 = nn.Linear(in_features=32, out_features=32)
        self.layer4 = nn.Linear(in_features=32, out_features=32)
        self.layer5 = nn.Linear(in_features=32, out_features=32)

        self.regressor = nn.Sequential(
            PMA(dim=32, num_heads=4, num_seeds=1),
        )

    def forward(self,
                t: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            t (torch.Tensor): Input tensor for time dimension.
            x (torch.Tensor): Input tensor for features.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.layer1(x)
        x = nn.ReLU()(x)
        x = x - torch.mean(x, 2, keepdim=True)
        x = self.layer2(x)

        x = nn.ReLU()(x)
        x = x - torch.mean(x, 2, keepdim=True)
        x = self.layer3(x)

        x = nn.ReLU()(x)
        x = x - torch.mean(x, 2, keepdim=True)
        x = self.layer4(x)
        x = nn.ReLU()(x)
        x = x - torch.mean(x, 2, keepdim=True)
        x = self.layer5(x)
        output = nn.ReLU()(x)

        output = self.regressor(output)
        output = torch.squeeze(output)

        output = self.time_layer2(t, output)
        output = nn.ReLU()(output)
        output = self.time_layer3(t, output)
        output = nn.ReLU()(output)
        output = self.time_layer4(t, output)
        output = nn.ReLU()(output)
        output = self.time_layer5(t, output)


        return output
import torch
import torch.nn as nn



class FBSNNNetwork(nn.Module):
    def __init__(
            self,
            layersize: list[int]
            ):
        """
        Initializes a neural network with multiple blocks.

        Args:
        - indim (int): input dimension
        - layersize (List[int]): list of sizes of hidden layers
        - outdim (int): output dimension
        """
        super(FBSNNNetwork, self).__init__()

        # initialize first set of layers

        self.n_layer = len(layersize) - 1
        # create input layer

        self.layers = nn.ModuleList([])
        # create hidden layers
        for i in range(len(layersize) - 1):
          self.layers.append(nn.Linear(layersize[i], layersize[i + 1]))

    def forward(
            self,
            x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the neural network.

        Args:
        - x (torch.Tensor): input tensor

        Returns:
        - torch.Tensor: output tensor
        """

        for i in range(self.n_layer):



            # apply layer to tensor
            x = self.layers[i](x)
            if i < self.n_layer - 1 :
              x =torch.sin(x)

        # apply output layer to tensor


        return x


class DeepBSDE(nn.Module):
    def __init__(
            self,
            layersize: list[int]
            ):
        """
        Initializes a neural network with multiple blocks.

        Args:
        - indim (int): input dimension
        - layersize (List[int]): list of sizes of hidden layers
        - outdim (int): output dimension
        """
        super(DeepBSDE, self).__init__()

        # initialize first set of layers

        self.n_layer = len(layersize) - 1
        # create input layer

        self.layers = nn.ModuleList([])
        self.batch_layer = nn.ModuleList([])
        # create hidden layers
        for i in range(len(layersize) - 1):
          self.layers.append(nn.Linear(layersize[i], layersize[i + 1] , bias=False))
          self.batch_layer.append(nn.BatchNorm1d(layersize[i] , eps=1e-06, momentum=0.01))


        self.batch_layer.append(nn.BatchNorm1d(layersize[-1] , eps=1e-06, momentum=0.01))

    def forward(
            self,
            x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the neural network.

        Args:
        - x (torch.Tensor): input tensor

        Returns:
        - torch.Tensor: output tensor
        """
        #x = self.batch_layer[0](x)
        for i in range(self.n_layer - 1):

            # apply layer to tensor
            x = self.layers[i](x)
            #x = self.batch_layer[i+1](x)


            x =nn.ReLU()(x)


        # apply output layer to tensor
        x = self.layers[-1](x)
        #x = self.batch_layer[-1](x)


        return x


class TimeDependentNetwork(nn.Module):
    def __init__(
            self,
            indim: int,
            layersize: list[int],
            outdim: int):
        """
        Initializes a neural network with multiple blocks.

        Args:
        - indim (int): input dimension
        - layersize (List[int]): list of sizes of hidden layers
        - outdim (int): output dimension
        """
        super(TimeDependentNetwork, self).__init__()

        # initialize first set of layers
        self.n_layer = len(layersize)
        self.layers = nn.ModuleList([])
        self.time_layer = nn.ModuleList([])
        self.batch_layer = nn.ModuleList([])

        # create input layer
        self.layers.append(Net1(indim, layersize[0]))
        self.time_layer.append(TimeNet(indim))
        self.batch_layer.append(nn.BatchNorm1d(indim , eps=1e-06, momentum=0.01))
        self.batch_layer.append(nn.BatchNorm1d(layersize[0],eps=1e-06, momentum=0.01))

        # create hidden layers
        for i in range(len(layersize) - 1):
            self.layers.append(Net1(layersize[i], layersize[i + 1]))
            self.time_layer.append(TimeNet(layersize[i]))
            self.batch_layer.append(nn.BatchNorm1d(layersize[i + 1] , eps=1e-06, momentum=0.01))

        # create output layer
        self.time_layer.append(TimeNet(outdim))
        self.linear = nn.Linear(layersize[-1], outdim)
        self.batch_layer.append(nn.BatchNorm1d(outdim , eps=1e-06, momentum=0.01))


    def forward(
            self,
            t: torch.Tensor,
            x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the neural network.

        Args:
        - t (torch.Tensor): tensor containing time information
        - x (torch.Tensor): input tensor

        Returns:
        - torch.Tensor: output tensor
        """
        for i in range(self.n_layer):
            # apply time-based weights to input tensor
            time = self.time_layer[i](torch.cat((t, t**2, t**3, torch.exp(t)), 1))
            x = self.batch_layer[i](x)


            x = x * (1 + time)

            # apply layer to tensor
            x = self.layers[i](x)

            x = torch.sin(x)

        # apply output layer to tensor

        x = self.linear(x)
        #x = self.batch_layer[-1](x)

        return x

class TimeDependentNetworkMonteCarlo(nn.Module):
    def __init__(
            self,
            indim: int,
            layersize: list[int],
            outdim: int,
            sigma: float):
        """
        Initializes a neural network with multiple blocks.

        Args:
        - indim (int): input dimension
        - layersize (List[int]): list of sizes of hidden layers
        - outdim (int): output dimension
        - sigma (float) : volatility
        """
        super(TimeDependentNetworkMonteCarlo, self).__init__()

        # initialize first set of layers
        self.n_layer = len(layersize)
        self.layers = nn.ModuleList([])
        self.time_layer = nn.ModuleList([])
        self.batch_layer = nn.ModuleList([])

        # create input layer
        self.layers.append(Net1(indim, layersize[0]))
        self.time_layer.append(TimeNet(indim))
        self.batch_layer.append(nn.BatchNorm1d(indim))
        self.batch_layer.append(nn.BatchNorm1d(layersize[0]))
        self.sigma = sigma

        # create hidden layers
        for i in range(len(layersize) - 1):
            self.layers.append(Net1(layersize[i], layersize[i + 1]))
            self.time_layer.append(TimeNet(layersize[i]))
            self.batch_layer.append(nn.BatchNorm1d(layersize[i + 1]))

        # create output layer
        self.time_layer.append(TimeNet(outdim))
        self.linear = nn.Linear(layersize[-1], outdim)

    def forward(
            self,
            t: torch.Tensor,
            x: torch.Tensor,
            y) -> torch.Tensor:
        """
        Passes the input through the neural network.

        Args:
        - t (torch.Tensor): tensor containing time information
        - x (torch.Tensor): input tensor

        Returns:
        - torch.Tensor: output tensor
        """
        x_prim = x

        for i in range(self.n_layer):
            # apply time-based weights to input tensor
            time = self.time_layer[i](torch.cat((t, t**2, t**3, torch.exp(t)), 1))

            x = x * (1 + time)

            # apply layer to tensor
            x = self.layers[i](x)
            x = nn.ReLU()(x)

        # apply output layer to tensor
        time = self.time_layer[self.n_layer](torch.cat((t, t**2, t**3, torch.exp(t)), 1))
        x = (self.sigma * x_prim * self.linear(x) + (1 - time) * y)

        return x