### Initial trial at MLE training of flows on toy data #
#
#

import itertools

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch import distributions
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from torch.nn.parameter import Parameter

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader

# load flows
from nflib.flows import MAF, NormalizingFlowModel, Invertible1x1Conv, ActNorm
from nflib.spline_flows import NSF_AR, NSF_CL

import os
os.chdir('../../')
from generateToyData import CustomSyntheticDatasetDensity, gen2Dspiral


# generate MoG data
dat = gen2Dspiral( n = 2500, radius=3.5) 

plt.scatter( dat[:,0], dat[:,1])

dset = CustomSyntheticDatasetDensity(dat.astype(np.float32) ) 
train_loader = DataLoader( dset, shuffle=True, batch_size=128 )

# define Flow model
prior = TransformedDistribution(Uniform(torch.zeros(2), torch.ones(2)), SigmoidTransform().inv) 

# MAF (with MADE net, so we get very fast density estimation)
#flows = [MAF(dim=2, parity=i%2) for i in range(4)]

nfs_flow = NSF_CL 
flows = [nfs_flow(dim=2, K=8, B=3, hidden_dim=16) for _ in range(4)]
convs = [Invertible1x1Conv(dim=2) for _ in flows]
norms = [ActNorm(dim=2) for _ in flows]
flows = list(itertools.chain(*zip(norms, convs, flows)))

# construct the model
model = NormalizingFlowModel(prior, flows)

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) # todo tune WD
print("number of params: ", sum(p.numel() for p in model.parameters()))

# run optimization
epochs = 250
loss_vals = []

model.train()
for e in range( epochs ):
	loss_val = 0
	for _, dat in enumerate( train_loader ):
		zs, prior_logprob, log_det = model( dat )
		logprob = prior_logprob + log_det
		loss = - torch.sum(logprob) # NLL

		#print(loss.item())
		loss_val += loss.item()

		# 
		model.zero_grad()
		optimizer.zero_grad()

		# compute gradients
		loss.backward()

		# update parameters
		optimizer.step()

	print('epoch {}/{} \tloss: {}'.format(e, epochs, loss_val))
	loss_vals.append( loss_val )


#plt.plot( range(len(loss_vals)), loss_vals)

# sample to see if it looks right:
x = model.sample(1000)[-1].detach().numpy()
plt.scatter( x[:,0], x[:,1])
plt.ion(); plt.show()

# compute the density over a grid
xvals = np.arange(-2.25, 2.25, .15)
yvals = np.arange(-2.25, 2.25, .15)

X, Y = np.meshgrid(xvals, yvals)

model.eval() 
Z = np.zeros( X.shape )
for i in range( X.shape[0] ):
	for j in range( X.shape[1] ):
		input_t = torch.Tensor([ [X[i,j], Y[i,j]], [1,1] ]) 
		zs, prior_logprob, log_det = model( input_t )
		Z[i,j] += ( (prior_logprob[0] + log_det[0]).item() )


plt.imshow( Z / Z.sum() )
plt.title('DEEN Paper Fig 1(a)')

