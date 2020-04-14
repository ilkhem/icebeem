### run TCL experiments
#
#

from data.imca import gen_TCL_data_ortho
from data.utils import to_one_hot
from metrics.mcc import mean_corr_coef
from models.tcl.tcl_wrapper_gpu import TCL_wrapper

data_dim = 5
data_segments = 10
n_layer = [2, 4]
n_obs_seg = [100, 200, 500, 1000, 2000]
n_sims = 10

results = {l: {n: [] for n in n_obs_seg} for l in n_layer}

for l in n_layer:
    for n in n_obs_seg:
        print('Running exp with L={} and n={}'.format(l, n))
        # generate some TCL data
        dat_all = gen_TCL_data_ortho(Ncomp=data_dim, Nsegment=data_segments, Nlayer=l, source='Gaussian', NsegmentObs=n,
                                     NonLin='leaky', negSlope=.2, Niter4condThresh=1e4)
        data = dat_all['obs']
        ut = to_one_hot(dat_all['labels'])[0]
        st = dat_all['source']

        # run iVAE 
        res_TCL = TCL_wrapper(sensor=data.T, label=dat_all['labels'],
                              list_hidden_nodes=[num_comp * 2] * (n_layer - 1) + [num_comp],
                              max_steps=stepDict[n_layer][0] * 2, max_steps_init=stepDict[n_layer][1])

        # store results
        results[l][n].append(mean_corr_coef(res_TCL[0].T, st))

# prepare output
Results = {
    'data_dim': data_dim,
    'data_segments': data_segments,
    'CorrelationCoef': results
}
