### run iVAE experiments
#
#

from models.ivae.IVAE_wrapper import * 

from metrics.solvehungarian import SolveHungarian

from data.nica import genTCLdataOrtho

import torch
import os 
torch.set_default_tensor_type('torch.cuda.FloatTensor')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# replace this later on 
# collect arguments
import sys
args = sys.argv 
data_dim      = 5 
data_segments = 10
n_layer       = [2, 4] 
n_obs_seg     = [100, 200, 500, 1000, 2000]
n_sims        = 10

results = {l:{n:[] for n in n_obs_seg} for l in n_layer}

for l in n_layer:
	for n in n_obs_seg:
		print('Running exp with L={} and n={}'.format(l,n))
		# generate some TCL data
		dat_all  = genTCLdataOrtho( Ncomp=data_dim, Nsegment=data_segments, Nlayer=l, source = 'Gaussian', NsegmentObs=n, NonLin='leaky', negSlope=.2, Niter4condThresh=1e4 )
		data     = dat_all['obs']
		ut       = to_one_hot( dat_all['labels'] )[0]
		st       = dat_all['source']

		# run iVAE 
		res_iVAE = IVAE_wrapper( X = data, U = to_one_hot(ut.argmax(1))[0], n_layers=n_layer+1, hidden_dim = data_dim*2, cuda=False, max_iter = 1e4 )

		# store results
		results[l][n].append( SolveHungarian( res_iVAE[0].detach().numpy(), st)[0] )


# prepare output
Results = {
	'data_dim'        : data_dim,
	'data_segments'   : data_segments,
	'CorrelationCoef' : results
}


#filename = 'TCL_iVAEresults_dim' + str(data_dim) + "_Segments" + str(data_segments)+ '.p'
#pickle.dump( Results, open(filename, 'wb'))
