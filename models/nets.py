import torch
import torch.nn.functional as F
from torch import nn


class LeafParam(nn.Module):
    """
    just ignores the input and outputs a parameter tensor
    todo maybe this exists in PyTorch somewhere?
    """

    def __init__(self, n):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1, n))

    def forward(self, x):
        return self.p.expand(x.size(0), self.p.size(1))


class PositionalEncoder(nn.Module):
    """
    Each dimension of the input gets expanded out with sins/coses
    to "carve" out the space. Useful in low-dimensional cases with
    tightly "curled up" data.
    """

    def __init__(self, freqs=(.5, 1, 2, 4, 8)):
        super().__init__()
        self.freqs = freqs

    def forward(self, x):
        sines = [torch.sin(x * f) for f in self.freqs]
        coses = [torch.cos(x * f) for f in self.freqs]
        out = torch.cat(sines + coses, dim=1)
        return out


class MLP4(nn.Module):
    """ a simple 4-layer MLP4 """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class PosEncMLP(nn.Module):
    """
    Position Encoded MLP4, where the first layer performs position encoding.
    Each dimension of the input gets transformed to len(freqs)*2 dimensions
    using a fixed transformation of sin/cos of given frequencies.
    """

    def __init__(self, nin, nout, nh, freqs=(.5, 1, 2, 4, 8)):
        super().__init__()
        self.net = nn.Sequential(
            PositionalEncoder(freqs),
            MLP4(nin * len(freqs) * 2, nout, nh),
        )

    def forward(self, x):
        return self.net(x)


class MLPlayer(nn.Module):
    """
    implement basic module for MLP

    note that this module keeps the dimensions fixed! will implement a mapping from a
    vector of dimension input_size to another vector of dimension input_size
    """

    def __init__(self, input_size, output_size=None, activation_function=nn.functional.relu, use_bn=False):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.activation_function = activation_function
        self.linear_layer = nn.Linear(input_size, output_size)
        self.use_bn = use_bn
        self.bn_layer = nn.BatchNorm1d(input_size)

    def forward(self, x):
        if self.use_bn:
            x = self.bn_layer(x)
        linear_act = self.linear_layer(x)
        H_x = self.activation_function(linear_act)
        return H_x


class MLP(nn.Module):
    """
    define a MLP network!
    """

    def __init__(self, input_size, hidden_size, n_layers, activation_function=F.relu):
        """
        Input:
         - input_size  : dimension of input data (e.g., 784 for MNIST)
         - hidden_size : size of hidden representations
         - n_layers    : number of hidden layers
        """
        super().__init__()

        output_size = 1  # because we approximating a log density, output should be scalar!

        self.activation_function = activation_function
        self.linear1st = nn.Linear(input_size, hidden_size)  # map from data dim to dimension of hidden units
        self.Layers = nn.ModuleList(
            [MLPlayer(hidden_size, activation_function=self.activation_function) for _ in range(n_layers)])
        self.linearLast = nn.Linear(hidden_size,
                                    output_size)  # map from dimension of hidden units to dimension of output

    def forward(self, x):
        """
        forward pass through resnet
        """
        x = self.linear1st(x)
        for _, current_layer in enumerate(self.Layers):
            x = current_layer(x)
        x = self.linearLast(x)
        return x


class MLP_general(nn.Module):
    """
    define a MLP network - this is a more general class than MLP4 above, allows for user to specify
    the dimensions at each layer of the network
    """

    def __init__(self, input_size, hidden_size, n_layers, output_size=None, activation_function=F.relu, use_bn=False):
        """
        Input:
         - input_size  : dimension of input data (e.g., 784 for MNIST)
         - hidden_size : list of hidden representations, one entry per layer
         - n_layers    : number of hidden layers
        """
        super().__init__()

        if output_size is None:
            output_size = 1  # because we approximating a log density, output should be scalar!

        self.use_bn = use_bn
        self.activation_function = activation_function
        self.linear1st = nn.Linear(input_size, hidden_size[0])  # map from data dim to dimension of hidden units
        self.Layers = nn.ModuleList([MLPlayer(hidden_size[i - 1], hidden_size[i],
                                              activation_function=self.activation_function, use_bn=self.use_bn) for i in
                                     range(1, n_layers)])
        self.linearLast = nn.Linear(hidden_size[-1],
                                    output_size)  # map from dimension of hidden units to dimension of output

    def forward(self, x):
        """
        forward pass through resnet
        """
        x = self.linear1st(x)
        for current_layer in self.Layers:
            x = current_layer(x)
        x = self.linearLast(x)
        return x


class smoothReLU(nn.Module):
    """
    smooth ReLU activation function
    """

    def __init__(self, beta=1):
        super().__init__()
        self.beta = 1

    def forward(self, x):
        return x / (1 + torch.exp(-self.beta * x))


class CleanMLP(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, output_size, activation='lrelu', batch_norm=False):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.activation = activation
        self.batch_norm = batch_norm

        if activation == 'lrelu':
            act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            act = nn.ReLU()
        else:
            raise ValueError('wrong activation')

        # construct model
        if n_hidden == 0:
            modules = [nn.Linear(input_size, output_size)]
        else:
            modules = [nn.Linear(input_size, hidden_size), act] + batch_norm * [nn.BatchNorm1d(hidden_size)]

        for i in range(n_hidden - 1):
            modules += [nn.Linear(hidden_size, hidden_size), act] + batch_norm * [nn.BatchNorm1d(hidden_size)]

        modules += [nn.Linear(hidden_size, output_size)]

        self.net = nn.Sequential(*modules)

    def forward(self, x, y=None):
        return self.net(x)
