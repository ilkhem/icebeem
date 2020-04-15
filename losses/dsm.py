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
