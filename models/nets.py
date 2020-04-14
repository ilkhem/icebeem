import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from torch import nn


class LeafParam(nn.Module):
    """
    just ignores the input and outputs a parameter tensor, lol
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


def compute_sigma_saremi(dat):
    """
    compute median, mean and max distance between two samples as
    suggested in Saremi & Hyvarinen (2019)
    """

    if dat.shape[0] > 5000:
        dat = dat[:5000, :]  # otherwise computing pairwise dist takes too long!

    D = pairwise_distances(dat)[np.triu_indices(dat.shape[0])] / (2 * np.sqrt(dat.shape[1]))

    return {'meanD': np.mean(D),
            'p15D': np.percentile(D, 15),
            'p05D': np.percentile(D, 5),
            'p25D': np.percentile(D, 25),
            'p50D': np.percentile(D, 50), 'maxD': np.max(D)}


class ConstrainedMLP(nn.Module):
    """
    a constrained MLP that satisfies assumptions of prop1
    """

    def __init__(self, sizes, use_bn=False):
        super().__init__()

        self._state = 0

        assert len(sizes) >= 2

        incr = True
        for i in range(len(sizes) - 1):
            if sizes[i] > sizes[i + 1]:
                incr = False
                break

        decr = True
        for i in range(len(sizes) - 1):
            if sizes[i] < sizes[i + 1]:
                decr = False
                break

        assert incr or decr

        input_dim = sizes[0]
        hidden_dim = sizes[1:-1]
        n_layers = len(sizes)
        output_dim = sizes[-1]

        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_bn = use_bn

        if self.n_layers == 1:
            _fc_list = [nn.Linear(input_dim, output_dim)]
        else:
            _fc_list = [nn.Linear(input_dim, hidden_dim[0])]
            if use_bn:
                _fc_list.append(nn.BatchNorm1d(hidden_dim[0]))
            _fc_list.append(nn.LeakyReLU(0.2))
            for i in range(1, n_layers - 1):
                _fc_list.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
                if use_bn:
                    _fc_list.append(nn.BatchNorm1d(hidden_dim[i]))
                _fc_list.append(nn.LeakyReLU(0.2))
            _fc_list.append(nn.Linear(hidden_dim[n_layers - 2], output_dim))
        self.fc = nn.ModuleList(_fc_list)

    def forward(self, x):
        for layer in self.fc:
            x = layer(x)
        return x

    def normalize_weights(self):
        # TODO: Is this the way spectral norm is supposed to be implemented??
        for layer in self.fc:
            layer = nn.utils.spectral_norm(layer)


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



class ConvolutionalFeatureExtractorMNIST(nn.Module):
    def __init__(self, output_size, n_lin_end=1):
        super().__init__()

        self.input_size = 784
        self.output_size = output_size
        self.hidden_size = 200
        self.n_lin_end = n_lin_end

        self.conv = nn.Sequential(
            # input is mnist image: 1x28x28
            nn.Conv2d(1, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.LeakyReLU(0.2, inplace=True),  # 32x14x14
            nn.Conv2d(32, 128, 4, 2, 1),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.LeakyReLU(0.2, inplace=True),  # 128x7x7
            nn.Conv2d(128, 512, 7, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.LeakyReLU(0.2, inplace=True),  # 512x1x1
            nn.Conv2d(512, 200, 1, 1, 0),  # 200x1x1
        )
        self.fc = nn.Sequential(
            *[nn.Linear(200, 200) for _ in range(n_lin_end - 1)],
            nn.Linear(200, output_size)
        )

    def forward(self, x):
        c = self.conv(x.view(-1, 1, 28, 28)).squeeze()  # make sure x is image for convolutional layer
        return self.fc(c)

class ARMLP(nn.Module):
    """ a 4-layer auto-regressive MLP, wrapper around MADE net """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = MADE(nin, [nh, nh, nh], nout, num_masks=1, natural_ordering=True)
        
    def forward(self, x):
        return self.net(x)


class MLP_general( nn.Module ):
  """
  define a MLP network - this is a more general class than MLP above, allows for user to specify
  the dimensions at each layer of the network
  """
  def __init__( self, input_size,  hidden_size, n_layers, output_size=None, activation_function = F.relu, use_bn=False ):
    """
    
    Input:
     - input_size  : dimension of input data (e.g., 784 for MNIST)
     - hidden_size : list of hidden representations, one entry per layer 
     - n_layers    : number of hidden layers 
     
    """
    super( MLP_general, self ).__init__()
    
    if output_size is None:
      output_size = 1 # because we approximating a log density, output should be scalar!

    self.use_bn = use_bn
    self.activation_function = activation_function
    self.linear1st = nn.Linear( input_size, hidden_size[0] ) # map from data dim to dimension of hidden units
    self.Layers = nn.ModuleList( [MLPlayer( hidden_size[i-1], hidden_size[i], activation_function=self.activation_function, use_bn=self.use_bn ) for i in range(1,n_layers) ] )
    self.linearLast = nn.Linear( hidden_size[-1], output_size ) # map from dimension of hidden units to dimension of output
    
  def forward( self, x ):
    """
    forward pass through resnet
    """
    x = self.linear1st( x )
    for current_layer in self.Layers :
      x = current_layer( x )
    x =  self.linearLast( x ) 
    return x
