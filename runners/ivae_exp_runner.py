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

data_dim = 5
data_segments = 40
n_layer = [2, 4]
n_obs_seg = [100, 200, 500, 1000, 2000]

results = {l: {n: [] for n in n_obs_seg} for l in n_layer}


def runiVAEexp(args):
    """run iVAE simulations"""
    nSims = args.nSims
    simulationMethod = args.method
    test = args.test
    for l in n_layer:
        for n in n_obs_seg:
            print('Running exp with L={} and n={}'.format(l, n))
            for seed in range(nSims):
                # generate data
                x, y, s = generate_synthetic_data(data_dim, data_segments, n, l, seed=seed,
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
        'data_segments': data_segments,
        'CorrelationCoef': results
    }

    return Results

# filename = 'TCL_iVAEresults_dim' + str(data_dim) + "_Segments" + str(data_segments)+ '.p'
# pickle.dump( Results, open(filename, 'wb'))
