### run TCL experiments
#
#

from data.imca import generate_synthetic_data
from metrics.mcc import mean_corr_coef
from models.tcl.tcl_wrapper_gpu import TCL_wrapper


def runTCLexp(args, config):
    """run TCL simulations"""
    stepDict = {1: [int(5e3), int(5e3)], 2: [int(1e4), int(1e4)], 3: [int(1e4), int(1e4)], 4: [int(1e4), int(1e4)],
                5: [int(1e4), int(1e4)]}

    data_dim = config.data_dim
    n_segments = config.n_segments
    n_layers = config.n_layers
    n_obs_per_seg = config.n_obs_per_seg

    results = {l: {n: [] for n in n_obs_per_seg} for l in n_layers}

    num_comp = data_dim

    nSims = args.nSims
    simulationMethod = args.method
    test = args.test

    for l in n_layers:
        for n in n_obs_per_seg:
            for seed in range(nSims):
                print('Running exp with L={} and n={}; seed={}'.format(l, n, seed))
                # generate data
                x, y, s = generate_synthetic_data(data_dim, n_segments, n, l, seed=seed,
                                                  simulationMethod=simulationMethod, one_hot_labels=False)
                # run TCL
                res_TCL = TCL_wrapper(sensor=x.T, label=y,
                                      list_hidden_nodes=[num_comp * 2] * (l - 1) + [num_comp],
                                      max_steps=stepDict[l][0] * 2, max_steps_init=stepDict[l][1])

                # store results
                from sklearn.decomposition import FastICA
                results[l][n].append(mean_corr_coef(FastICA().fit_transform(res_TCL[0].T), s))

    # prepare output
    Results = {
        'data_dim': data_dim,
        'data_segments': n_segments,
        'CorrelationCoef': results
    }

    return Results
