### conditional dsm objective 
#
# this code is adapted from: https://github.com/ermongroup/ncsn/
#

import torch
import torch.autograd as autograd


def dsm(energy_net, samples, sigma=1):
    samples.requires_grad_(True)
    vector = torch.randn_like(samples) * sigma
    perturbed_inputs = samples + vector
    logp = -energy_net(perturbed_inputs)
    dlogp = sigma ** 2 * autograd.grad(logp.sum(), perturbed_inputs, create_graph=True)[0]
    kernel = vector
    loss = torch.norm(dlogp + kernel, dim=-1) ** 2
    loss = loss.mean() / 2.

    return loss


def cdsm(energy_net, samples, conditions, sigma=1.):
    """
    Conditional denoising score matching
    :param energy_net: an energy network that takes x and y as input and outputs energy of shape (batch_size,)
    :param samples: values of dependent variable x
    :param conditions: values of conditioning variable y
    :param sigma: noise level for dsm
    :return: cdsm loss of shape (batch_size,)
    """
    samples.requires_grad_(True)
    vector = torch.randn_like(samples) * sigma
    perturbed_inputs = samples + vector
    logp = -energy_net(perturbed_inputs, conditions)
    assert logp.ndim == 1
    dlogp = sigma ** 2 * autograd.grad(logp.sum(), perturbed_inputs, create_graph=True)[0]
    kernel = vector
    loss = torch.norm(dlogp + kernel, dim=-1) ** 2
    loss = loss.mean() / 2.
    return loss


def conditional_dsm(energy_net, samples, segLabels, energy_net_final_layer, sigma=1):
    samples.requires_grad_(True)
    vector = torch.randn_like(samples) * sigma
    perturbed_inputs = samples + vector

    d = samples.shape[-1]

    # apply conditioning
    logp = -energy_net(perturbed_inputs).view(-1, d * d)
    logp = torch.mm(logp, energy_net_final_layer)
    # take only relevant segment energy
    logp = logp[segLabels]

    dlogp = sigma ** 2 * autograd.grad(logp.sum(), perturbed_inputs, create_graph=True)[0]
    kernel = vector
    loss = torch.norm(dlogp + kernel, dim=-1) ** 2
    loss = loss.mean() / 2.

    return loss


def dsm_score_estimation(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)

    return loss
