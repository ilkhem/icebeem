import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import FastICA
from torch.distributions import Uniform, TransformedDistribution, SigmoidTransform

from .fce import ConditionalFCE
from .nets import MLP_general
from .nflib.flows import NormalizingFlowModel, Invertible1x1Conv, ActNorm
from .nflib.spline_flows import NSF_AR


def ICEBEEM_wrapper(X, Y, ebm_hidden_size, n_layers_ebm, n_layers_flow, lr_flow, lr_ebm, seed,
                    ckpt_file='icebeem.pt', test=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_dim = X.shape[1]

    model_ebm = MLP_general(input_size=data_dim, hidden_size=[ebm_hidden_size] * n_layers_ebm,
                            n_layers=n_layers_ebm, output_size=data_dim, use_bn=True,
                            activation_function=F.leaky_relu)

    prior = TransformedDistribution(Uniform(torch.zeros(data_dim), torch.ones(data_dim)),
                                    SigmoidTransform().inv)
    nfs_flow = NSF_AR
    flows = [nfs_flow(dim=data_dim, K=8, B=3, hidden_dim=16) for _ in range(n_layers_flow)]
    convs = [Invertible1x1Conv(dim=data_dim) for _ in flows]
    norms = [ActNorm(dim=data_dim) for _ in flows]
    flows = list(itertools.chain(*zip(norms, convs, flows)))
    # construct the model
    model_flow = NormalizingFlowModel(prior, flows)

    pretrain_flow = True
    augment_ebm = True

    # instantiate ebmFCE object
    fce_ = ConditionalFCE(data=X.astype(np.float32), segments=Y.astype(np.float32),
                          energy_MLP=model_ebm, flow_model=model_flow, verbose=False)

    init_ckpt_file = os.path.splitext(ckpt_file)[0] + '_0' + os.path.splitext(ckpt_file)[1]
    if not test:
        if pretrain_flow:
            # print('pretraining flow model..')
            fce_.pretrain_flow_model(epochs=1, lr=1e-4)
            # print('pretraining done.')

        # first we pretrain the final layer of EBM model (this is g(y) as it depends on segments)
        fce_.train_ebm_fce(epochs=15, augment=augment_ebm, finalLayerOnly=True, cutoff=.5)

        # then train full EBM via NCE with flow contrastive noise:
        fce_.train_ebm_fce(epochs=50, augment=augment_ebm, cutoff=.5, useVAT=False)

        torch.save({'ebm_mlp': fce_.energy_MLP.state_dict(),
                    'ebm_finalLayer': fce_.ebm_finalLayer,
                    'flow': fce_.flow_model.state_dict()}, init_ckpt_file)
    else:
        state = torch.load(init_ckpt_file, map_location=fce_.device)
        fce_.energy_MLP.load_state_dict(state['ebm_mlp'])
        fce_.ebm_finalLayer = state['ebm_finalLayer']
        fce_.flow_model.load_stat_dict(state['flow'])

    # evaluate recovery of latents
    recov = fce_.unmixSamples(X, modelChoice='ebm')
    source_est_ica = FastICA().fit_transform((recov))
    recov_sources = [source_est_ica]

    # iterate between updating noise and tuning the EBM
    eps = .025
    for iter_ in range(3):
        mid_ckpt_file = os.path.splitext(ckpt_file)[0] + '_' + str(iter_ + 1) + os.path.splitext(ckpt_file)[1]
        if not test:
            # update flow model:
            fce_.train_flow_fce(epochs=5, objConstant=-1., cutoff=.5 - eps, lr=lr_flow)
            # update energy based model:
            fce_.train_ebm_fce(epochs=50, augment=augment_ebm, cutoff=.5 + eps, lr=lr_ebm, useVAT=False)

            torch.save({'ebm_mlp': fce_.energy_MLP.state_dict(),
                        'ebm_finalLayer': fce_.ebm_finalLayer,
                        'flow': fce_.flow_model.state_dict()}, mid_ckpt_file)
        else:
            state = torch.load(mid_ckpt_file, map_location=fce_.device)
            fce_.energy_MLP.load_state_dict(state['ebm_mlp'])
            fce_.ebm_finalLayer = state['ebm_finalLayer']
            fce_.flow_model.load_stat_dict(state['flow'])

        # evaluate recovery of latents
        recov = fce_.unmixSamples(X, modelChoice='ebm')
        source_est_ica = FastICA().fit_transform((recov))
        recov_sources.append(source_est_ica)

    return recov_sources
