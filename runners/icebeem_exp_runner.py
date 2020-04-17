### run ICE-BeeM experiments (implemented here using FCE)
#
#
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import FastICA
from torch.distributions import Uniform, TransformedDistribution, SigmoidTransform

from data.imca import generate_synthetic_data
from metrics.mcc import mean_corr_coef
from models.icebeem_fce import ebmFCEsegments
from models.nets import MLP_general
from models.nflib.flows import NormalizingFlowModel, Invertible1x1Conv, ActNorm
from models.nflib.spline_flows import NSF_AR

torch.set_default_tensor_type('torch.cuda.FloatTensor')


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def runICEBeeMexp(args, config):
    """run ICE-BeeM simulations"""
    data_dim = config.data_dim
    n_segments = config.n_segments
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg

    lr_flow = config.icebeem.lr_flow
    lr_ebm = config.icebeem.lr_ebm
    n_layers_flow = config.icebeem.n_layers_flow
    ebm_hidden_size = config.icebeem.ebm_hidden_size

    results = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    nSims = args.nSims
    simulationMethod = args.method
    test = args.test

    for l in n_layers:
        n_layers_ebm = l + 1
        for n in n_obs_per_seg:
            print('Running exp with L={} and n={}'.format(l, n))
            for seed in range(nSims):
                # generate data
                data, ut, st = generate_synthetic_data(data_dim, n_segments, n, l, seed=seed,
                                                       simulationMethod=simulationMethod, one_hot_labels=True)

                # define and run ICEBEEM
                model_ebm = MLP_general(input_size=data_dim, hidden_size=[ebm_hidden_size] * n_layers_ebm,
                                        n_layers=n_layers_ebm, output_size=data_dim, use_bn=True,
                                        activation_function=F.leaky_relu)
                # model_ebm = CleanMLP( input_size=data_dim, n_hidden=n_layers_ebm, hidden_size=data_dim*2, output_size=data_dim, batch_norm=True)

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
                fce_ = ebmFCEsegments(data=data.astype(np.float32), segments=ut.astype(np.float32),
                                      energy_MLP=model_ebm, flow_model=model_flow, verbose=False)

                if pretrain_flow:
                    # print('pretraining flow model..')
                    fce_.pretrain_flow_model(epochs=1, lr=1e-4)
                    # print('pretraining done.')

                # first we pretrain the final layer of EBM model (this is g(y) as it depends on segments)
                fce_.train_ebm_fce(epochs=15, augment=augment_ebm, finalLayerOnly=True, cutoff=.5)

                # then train full EBM via NCE with flow contrastive noise:
                fce_.train_ebm_fce(epochs=50, augment=augment_ebm, cutoff=.5, useVAT=False)

                # evaluate recovery of latents
                recov = fce_.unmixSamples(data, modelChoice='ebm')
                source_est_ica = FastICA().fit_transform((recov))
                recov_sources = [source_est_ica]

                # iterate between updating noise and tuning the EBM
                eps = .025
                for iter_ in range(3):
                    # update flow model:
                    fce_.train_flow_fce(epochs=5, objConstant=-1., cutoff=.5 - eps, lr=lr_flow)
                    # update energy based model:
                    fce_.train_ebm_fce(epochs=50, augment=augment_ebm, cutoff=.5 + eps, lr=lr_ebm, useVAT=False)

                    # evaluate recovery of latents
                    recov = fce_.unmixSamples(data, modelChoice='ebm')
                    source_est_ica = FastICA().fit_transform((recov))
                    recov_sources.append(source_est_ica)

                # store results
                results[l][n].append(np.max([mean_corr_coef(x, st) for x in recov_sources]))

                print(np.max([mean_corr_coef(x, st) for x in recov_sources]))

    # prepare output
    Results = {
        'data_dim': data_dim,
        'data_segments': n_segments,
        'CorrelationCoef': results
    }

    return Results
