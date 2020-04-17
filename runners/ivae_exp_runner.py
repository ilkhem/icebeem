### run iVAE experiments
#
#

import os

import torch

from data.imca import generate_synthetic_data
from metrics.mcc import mean_corr_coef
from models.ivae.ivae_wrapper import IVAE_wrapper

torch.set_default_tensor_type('torch.cuda.FloatTensor')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def runiVAEexp(args, config):
    """run iVAE simulations"""
    data_dim = config.data_dim
    n_segments = config.n_segments
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg

    results = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    nSims = args.nSims
    simulationMethod = args.method
    test = args.test
    for l in n_layers:
        for n in n_obs_per_seg:
            for seed in range(nSims):
                print('Running exp with L={} and n={}; seed={}'.format(l, n, seed))
                # generate data
                x, y, s = generate_synthetic_data(data_dim, n_segments, n, l, seed=seed,
                                                  simulationMethod=simulationMethod, one_hot_labels=True)
                # run iVAE
                ckpt_file = 'ivae_l{}_n{}_s{}.pt'.format(l, n, seed)
                res_iVAE = IVAE_wrapper(X=x, U=y, n_layers=l + 1, hidden_dim=data_dim * 2,
                                        cuda=False, max_iter=1e5, ckpt_file=ckpt_file, test=test)

                # store results
                results[l][n].append(mean_corr_coef(res_iVAE[0].detach().numpy(), s))

    # prepare output
    Results = {
        'data_dim': data_dim,
        'data_segments': n_segments,
        'CorrelationCoef': results
    }

    return Results

# filename = 'TCL_iVAEresults_dim' + str(data_dim) + "_Segments" + str(data_segments)+ '.p'
# pickle.dump( Results, open(filename, 'wb'))
