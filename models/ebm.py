import contextlib
import torch
import torch.nn.functional as F
from torch import nn

from .nets import CleanMLP


class UnnormalizedConditialEBM(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, output_size, condition_size, activation='lrelu'):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.cond_size = condition_size
        self.n_hidden = n_hidden
        self.activation = activation

        self.f = CleanMLP(input_size, hidden_size, n_hidden, output_size, activation=activation)
        self.g = nn.Linear(condition_size, output_size, bias=False)

    def log_pdf(self, x, y, augment=True, positive=False):
        fx, gy = self.forward(x, y)

        if positive:
            fx = F.relu(fx)
            gy = F.relu(gy)

        if augment:
            return torch.einsum('bi,bi->b', [fx, gy]) + torch.einsum('bi,bi->b', [fx.pow(2), gy.pow(2)])

        else:
            return torch.einsum('bi,bi->b', [fx, gy])

    def forward(self, x, y):
        fx = self.f(x)
        gy = self.g(y)
        return fx, gy


class ModularUnnormalizedConditionalEBM(nn.Module):
    def __init__(self, f_net, g_net):
        super().__init__()

        assert f_net.output_size == g_net.out_features

        self.input_size = f_net.input_size
        self.output_size = f_net.output_size
        self.cond_size = g_net.in_features

        self.f = f_net
        self.g = g_net

    def forward(self, x, y):
        return self.f(x), self.g(y)

    def log_pdf(self, x, y, augment=True, positive=False):
        fx, gy = self.forward(x, y)

        if positive:
            fx = F.relu(fx)
            gy = F.relu(gy)

        if augment:
            return torch.einsum('bi,bi->b', [fx, gy]) + torch.einsum('bi,bi->b', [fx.pow(2), gy.pow(2)])

        else:
            return torch.einsum('bi,bi->b', [fx, gy])


class ConditionalEBM(UnnormalizedConditialEBM):
    def __init__(self, input_size, hidden_size, n_hidden, output_size, condition_size, activation='lrelu'):
        super().__init__(input_size, hidden_size, n_hidden, output_size, condition_size, activation)

        self.log_norm = nn.Parameter(torch.randn(1) - 5, requires_grad=True)

    def log_pdf(self, x, y, augment=True, positive=False):
        return super().log_pdf(x, y, augment, positive) + self.log_norm


class ModularConditionalEBM(ModularUnnormalizedConditionalEBM):
    def __init__(self, f_net, g_net):
        super().__init__(f_net, g_net)

        self.log_norm = nn.Parameter(torch.randn(1) - 5, requires_grad=True)

    def log_pdf(self, x, y, augment=True, positive=False):
        return super().log_pdf(x, y, augment, positive) + self.log_norm


### Virtual adversarial regularization loss
#
#
# this code has been shamelessly taken from:
#  https://raw.githubusercontent.com/lyakaap/VAT-pytorch/master/vat.py
#
###


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
