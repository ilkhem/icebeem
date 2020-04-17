### run TCL experiments
#
#

from data.imca import generate_synthetic_data
from metrics.mcc import mean_corr_coef
from models.tcl.tcl_wrapper_gpu import TCL_wrapper

data_dim = 5
data_segments = 8
n_layer = [2, 4]
n_obs_seg = [100, 200, 500, 1000, 2000]

stepDict = {1: [int(5e3), int(5e3)], 2: [int(1e4), int(1e4)], 3: [int(1e4), int(1e4)], 4: [int(1e4), int(1e4)],
            5: [int(1e4), int(1e4)]}


def runTCLexp(nSims=10, simulationMethod='TCL'):
    """run TCL simulations"""

    results = {l: {n: [] for n in n_obs_seg} for l in n_layer}
    num_comp = data_dim

    for l in n_layer:
        for n in n_obs_seg:
            print('Running exp with L={} and n={}'.format(l, n))
            # generate some TCL data
            for seed in range(nSims):
                x, y, s = generate_synthetic_data(data_dim, data_segments, n, l, seed=seed,
                                                  simulationMethod=simulationMethod, one_hot_labels=False)
                # run iVAE
                res_TCL = TCL_wrapper(sensor=x.T, label=y,
                                      list_hidden_nodes=[num_comp * 2] * (l - 1) + [num_comp],
                                      max_steps=stepDict[l][0] * 2, max_steps_init=stepDict[l][1])

                # store results
                from sklearn.decomposition import FastICA
                results[l][n].append(mean_corr_coef(FastICA().fit_transform(res_TCL[0].T), s))

    # prepare output
    Results = {
        'data_dim': data_dim,
        'data_segments': data_segments,
        'CorrelationCoef': results
    }

    return Results
