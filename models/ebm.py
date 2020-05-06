import torch
import torch.nn.functional as F
from torch import nn

from .nets import CleanMLP


class UnnormalizedConditialEBM(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, output_size, condition_size, activation='lrelu',
                 augment=False, positive=False):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.cond_size = condition_size
        self.n_hidden = n_hidden
        self.activation = activation
        self.augment = augment
        self.positive = positive

        self.f = CleanMLP(input_size, hidden_size, n_hidden, output_size, activation=activation)
        self.g = nn.Linear(condition_size, output_size, bias=False)

    def forward(self, x, y):
        fx, gy = self.f(x).view(-1, self.output_size), self.g(y)

        if self.positive:
            fx = F.relu(fx)
            gy = F.relu(gy)

        if self.augment:
            return torch.einsum('bi,bi->b', [fx, gy]) + torch.einsum('bi,bi->b', [fx.pow(2), gy.pow(2)])

        else:
            return torch.einsum('bi,bi->b', [fx, gy])


class ModularUnnormalizedConditionalEBM(nn.Module):
    def __init__(self, f_net, g_net, augment=False, positive=False):
        super().__init__()

        assert f_net.output_size == g_net.output_size

        self.input_size = f_net.input_size
        self.output_size = f_net.output_size
        self.cond_size = g_net.input_size
        self.augment = augment
        self.positive = positive

        self.f = f_net
        self.g = g_net

    def forward(self, x, y):
        fx, gy = self.f(x).view(-1, self.output_size), self.g(y)

        if self.positive:
            fx = F.relu(fx)
            gy = F.relu(gy)

        if self.augment:
            return torch.einsum('bi,bi->b', [fx, gy]) + torch.einsum('bi,bi->b', [fx.pow(2), gy.pow(2)])

        else:
            return torch.einsum('bi,bi->b', [fx, gy])


class ConditionalEBM(UnnormalizedConditialEBM):
    def __init__(self, input_size, hidden_size, n_hidden, output_size, condition_size, activation='lrelu'):
        super().__init__(input_size, hidden_size, n_hidden, output_size, condition_size, activation)

        self.log_norm = nn.Parameter(torch.randn(1) - 5, requires_grad=True)

    def forward(self, x, y, augment=True, positive=False):
        return super().forward(x, y, augment, positive) + self.log_norm


class ModularConditionalEBM(ModularUnnormalizedConditionalEBM):
    def __init__(self, f_net, g_net):
        super().__init__(f_net, g_net)

        self.log_norm = nn.Parameter(torch.randn(1) - 5, requires_grad=True)

    def forward(self, x, y, augment=True, positive=False):
        return super().forward(x, y, augment, positive) + self.log_norm


class UnnormalizedEBM(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, output_size, activation='lrelu'):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.activation = activation

        self.f = CleanMLP(input_size, hidden_size, n_hidden, output_size, activation=activation)
        self.g = torch.ones(output_size)

    def forward(self, x, y=None):
        fx = self.f(x).view(-1, self.output_size)
        return torch.einsum('bi,i->b', [fx, self.g])


class ModularUnnormalizedEBM(nn.Module):
    def __init__(self, f_net):
        super().__init__()

        self.input_size = f_net.input_size
        self.output_size = f_net.output_size

        self.f = f_net
        self.g = torch.ones(self.output_size)

    def forward(self, x, y=None):
        fx = self.f(x).view(-1, self.output_size)
        return torch.einsum('bi,i->b', [fx, self.g])


class EBM(UnnormalizedEBM):
    def __init__(self, input_size, hidden_size, n_hidden, output_size, activation='lrelu'):
        super().__init__(input_size, hidden_size, n_hidden, output_size, activation)

        self.log_norm = nn.Parameter(torch.randn(1) - 5, requires_grad=True)

    def forward(self, x, y=None):
        return super().forward(x, y) + self.log_norm


class ModularEBM(ModularUnnormalizedEBM):
    def __init__(self, f_net):
        super().__init__(f_net)

        self.log_norm = nn.Parameter(torch.randn(1) - 5, requires_grad=True)

    def forward(self, x, y=None):
        return super().forward(x, y) + self.log_norm
