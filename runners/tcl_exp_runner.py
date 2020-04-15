### run TCL experiments
#
#
import numpy as np
from scipy.stats import random_correlation

from data.imca import gen_TCL_data_ortho, gen_IMCA_data
from data.utils import to_one_hot
from metrics.mcc import mean_corr_coef
from models.tcl.tcl_wrapper_gpu import TCL_wrapper

data_dim = 5
data_segments = 8
n_layer = [2, 4]
n_obs_seg = [100, 200, 500, 1000, 2000]

stepDict = {1: [int(5e3), int(5e3)], 2: [int(1e4), int(1e4)], 3: [int(1e4), int(1e4)], 4: [int(1e4), int(1e4)],
            5: [int(1e4), int(1e4)]}

def runTCLexp( nSims = 10, simulationMethod='TCL'):
    """run TCL simulations"""

    results = {l: {n: [] for n in n_obs_seg} for l in n_layer}
    num_comp = data_dim

    for l in n_layer:
        for n in n_obs_seg:
            print('Running exp with L={} and n={}'.format(l, n))
            # generate some TCL data
            for _ in range(nSims):
                if simulationMethod=='TCL':
                    dat_all = gen_TCL_data_ortho(Ncomp=data_dim, Nsegment=data_segments, Nlayer=l, source='Gaussian', NsegmentObs=n,
                                                 NonLin='leaky', negSlope=.2, Niter4condThresh=1e4)
                    data = dat_all['obs']
                    ut = to_one_hot(dat_all['labels'])[0]
                    st = dat_all['source']
                else:
                    baseEvals  = np.random.rand(data_dim)
                    baseEvals /= (.5 * baseEvals.sum() )
                    baseCov    = random_correlation.rvs( baseEvals )

                    dat_all  = gen_IMCA_data(Ncomp=data_dim, Nsegment=data_segments, Nlayer=n_layer,
                               NsegmentObs=n_obs_seg, NonLin='leaky',
                               negSlope=.2, Niter4condThresh=1e4,
                               BaseCovariance = baseCov)
                    data     = dat_all['obs']
                    ut       = to_one_hot( dat_all['labels'] )[0]
                    st       = dat_all['source']


                # run iVAE
                res_TCL = TCL_wrapper(sensor=data.T, label=dat_all['labels'],
                                      list_hidden_nodes=[num_comp * 2] * (l - 1) + [num_comp],
                                      max_steps=stepDict[l][0] * 2, max_steps_init=stepDict[l][1])

                # store results
                from sklearn.decomposition import FastICA
                results[l][n].append(mean_corr_coef( FastICA().fit_transform( res_TCL[0].T ), st))

    # prepare output
    Results = {
        'data_dim': data_dim,
        'data_segments': data_segments,
        'CorrelationCoef': results
    }

    return Results


