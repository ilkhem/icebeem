import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .spline_flows import unconstrained_RQS
from ..nets import MLP4


class Invertible1x1Conv(nn.Module):
    """
    As introduced in Glow paper.
    """

    def __init__(self, dim, device='cpu', condition_size=0):
        super().__init__()
        self.conditional = True  # forward backward unchanged when conditional, thus always True
        self.cond_size = condition_size  # for compatibility with the rest of the flow who have this attribute
        self.dim = dim
        self.device = device
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim).to(self.device))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P.to(self.device)  # remains fixed during optimization
        self.L = nn.Parameter(L).to(self.device)  # lower triangular portion
        self.S = nn.Parameter(U.diag()).to(self.device)  # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)).to(self.device)  # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim).to(self.device))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x, y=None):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def backward(self, z, y=None):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det


class NSF_CL(nn.Module):
    """ Neural spline flow, coupling layer, [Durkan et al. 2019] """

    def __init__(self, dim, K=5, B=3, hidden_dim=8, base_network=MLP4, device='cpu', condition_size=0):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.device = device
        self.conditional = condition_size != 0
        self.cond_size = condition_size
        self.f1 = base_network(dim // 2 + self.cond_size, (3 * K - 1) * dim // 2, hidden_dim).to(self.device)
        self.f2 = base_network(dim // 2 + self.cond_size, (3 * K - 1) * dim // 2, hidden_dim).to(self.device)

    def forward(self, x, y=None):
        log_det = torch.zeros(x.shape[0]).to(self.device)
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]
        if y is not None and self.conditional:
            lower_y = torch.cat([lower, y], dim=1)
        else:
            lower_y = lower
        out = self.f1(lower_y).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(upper, W, H, D, inverse=False, tail_bound=self.B, device=self.device)
        log_det += torch.sum(ld, dim=1)
        if y is not None and self.conditional:
            upper_y = torch.cat([upper, y], dim=1)
        else:
            upper_y = upper
        out = self.f2(upper_y).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=False, tail_bound=self.B, device=self.device)
        log_det += torch.sum(ld, dim=1)
        return torch.cat([lower, upper], dim=1), log_det

    def backward(self, z, y=None):
        log_det = torch.zeros(z.shape[0]).to(self.device)
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        if y is not None and self.conditional:
            upper_y = torch.cat([upper, y], dim=1)
        else:
            upper_y = upper
        out = self.f2(upper_y).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=True, tail_bound=self.B, device=self.device)
        log_det += torch.sum(ld, dim=1)
        if y is not None and self.conditional:
            lower_y = torch.cat([lower, y], dim=1)
        else:
            lower_y = lower
        out = self.f1(lower_y).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(upper, W, H, D, inverse=True, tail_bound=self.B, device=self.device)
        log_det += torch.sum(ld, dim=1)
        return torch.cat([lower, upper], dim=1), log_det


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows, device='cpu'):
        super().__init__()
        self.device = device
        self.flows = nn.ModuleList(flows)

        # the normalizing flow is conditional only if all its flows are conditional!
        self.conditional = True
        self.cond_size = 0
        for flow in self.flows:
            if not hasattr(flow, 'conditional'):
                self.conditional = False
            else:
                if not flow.conditional:
                    self.conditional = False
                if flow.cond_size > 0:
                    self.cond_size = flow.cond_size

    def forward(self, x, y=None):
        m, _ = x.shape
        log_det = torch.zeros(m).to(self.device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x, y)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z, y=None):
        m, _ = z.shape
        log_det = torch.zeros(m).to(self.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z, y)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior, flows, device='cpu'):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows, device=device)
        self.device = device
        self.conditional = self.flow.conditional  # conditional if its flow is conditional
        self.cond_size = self.flow.cond_size

    def forward(self, x, y=None):
        zs, log_det = self.flow.forward(x, y)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z, y=None):
        xs, log_det = self.flow.backward(z, y)
        return xs, log_det

    def log_pdf(self, x, y=None):
        zs, prior_logprob, log_det = self.forward(x, y)
        flow_logdensity = (prior_logprob + log_det)
        return flow_logdensity

    def sample(self, n_samples, cond_size=None):

        self.eval()
        # conditional model
        if self.conditional and cond_size is not None:
            # generate n_samples in total
            n_samples_per_cond = n_samples // cond_size  # nbr of samples per condition
            n_samples_rem = n_samples % cond_size  # remaining of division to have toatal n_samples samples
            # THIS WORKS ONLY FOR DISCRETE CONDITONING VARIABLES
            samples = []
            labels = torch.eye(cond_size).to(self.device)
            ret_labels = []
            ret_log_probs = []

            if n_samples_per_cond > 0:
                for i in range(cond_size):
                    # sample model base distribution and run through backward model to sample data space
                    u = self.prior.sample((n_samples_per_cond,))
                    labels_i = labels[i].expand(n_samples_per_cond, -1)

                    sample, _ = self.backward(u, labels_i)
                    sample = sample[-1]
                    log_probs = self.log_pdf(sample, labels_i)
                    samples.append(sample)
                    ret_log_probs.append(log_probs)
                    ret_labels.append(labels_i)

            # sample the remaining samples from a RANDOM condition
            if n_samples_rem > 0:
                j = np.random.randint(cond_size)
                u = self.prior.sample((n_samples_rem,))
                labels_j = labels[j].expand(n_samples_rem, -1)
                sample, _ = self.backward(u, labels_j)
                sample = sample[-1]
                log_probs = self.log_pdf(sample, labels_j)
                samples.append(sample)
                ret_log_probs.append(log_probs)
                ret_labels.append(labels_j)

            samples = torch.cat(samples, dim=0)
            ret_labels = torch.cat(ret_labels, dim=0)
            ret_log_probs = torch.cat(ret_log_probs, dim=0)

        # unconditional model
        else:
            u = self.prior.sample((n_samples,))
            samples, _ = self.backward(u)
            samples = samples[-1]
            log_probs = self.log_pdf(samples)
            ret_labels = torch.tensor([])
            ret_log_probs = log_probs

        # convert and save images
        lam = 1e-6
        samples = (torch.sigmoid(samples) - lam) / (1 - 2 * lam)

        return samples, ret_labels, ret_log_probs
