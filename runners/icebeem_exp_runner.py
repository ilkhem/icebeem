### run ICE-BeeM experiments (implemented here using FCE)
#
#

import numpy as np
import torch

from data.imca import generate_synthetic_data
from metrics.mcc import mean_corr_coef
from models.icebeem_wrapper import ICEBEEM_wrapper

torch.set_default_tensor_type('torch.cuda.FloatTensor')


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'



def runICEBeeMexp(args, config):
    """run ICE-BeeM simulations"""
    data_dim = config.data_dim
    n_segments = config.n_segments
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg
    data_seed = config.data_seed

    lr_flow = config.icebeem.lr_flow
    lr_ebm = config.icebeem.lr_ebm
    n_layers_flow = config.icebeem.n_layers_flow
    ebm_hidden_size = config.icebeem.ebm_hidden_size

    results = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    nSims = args.nSims
    simulationMethod = args.dataset
    test = args.test

    for l in n_layers:
        for n in n_obs_per_seg:
            x, y, s = generate_synthetic_data(data_dim, n_segments, n, l, seed=data_seed,
                                              simulationMethod=simulationMethod, one_hot_labels=True)
            for seed in range(nSims):
                print('Running exp with L={} and n={}; seed={}'.format(l, n, seed))
                # generate data

                n_layers_ebm = l + 1
                recov_sources = ICEBEEM_wrapper(X=x, Y=y, ebm_hidden_size=ebm_hidden_size,
                                                n_layers_ebm=n_layers_ebm, n_layers_flow=n_layers_flow,
                                                lr_flow=lr_flow, lr_ebm=lr_ebm, seed=seed)

                # store results
                results[l][n].append(np.max([mean_corr_coef(z, s) for z in recov_sources]))
                print(np.max([mean_corr_coef(z, s) for z in recov_sources]))

    # prepare output
    Results = {
        'data_dim': data_dim,
        'data_segments': n_segments,
        'CorrelationCoef': results
    }

    return Results
