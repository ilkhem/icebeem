### Pytorch implementation of training EBMs via FCE 
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
from torch.utils.data import Dataset
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# import VAT loss
from models.ebm import UnnormalizedConditialEBM, ConditionalEBM, VATLoss

# load flows - will serve as noise distribution in FCE
from models.nflib.flows import MAF, NormalizingFlowModel, Invertible1x1Conv, ActNorm
from models.nflib.spline_flows import NSF_AR, NSF_CL
#from models.flows import MAF, NormalizingFlowModel, Invertible1x1Conv, ActNorm
#from models.spline_flows import NSF_AR, NSF_CL

# this should be removed later to use Ilyes' implementation of CBM
from models.nets import CleanMLP, MLP_general

# ------------------------
# define helper functions
# ------------------------

class CustomSyntheticDatasetDensity(Dataset):
	def __init__(self, X, device='cpu'):
		self.device = device
		self.x = torch.from_numpy(X).to(device)
		self.len = self.x.shape[0]
		self.data_dim = self.x.shape[1]
		print('data loaded on {}'.format(self.x.device))

	def get_dims(self):
		return self.data_dim

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.x[index]

	def get_metadata(self):
		return {
			'n': self.len,
			'data_dim': self.data_dim,
			}    


class CustomFCEdataset(Dataset):
	def __init__(self, X, Y, device='cpu'):
		self.device = device
		self.x = torch.from_numpy(X).to(device)
		self.y = torch.from_numpy(Y).to(device)
		self.len = self.x.shape[0]
		self.data_dim = self.x.shape[1]
		#print('data loaded on {}'.format(self.x.device))

	def get_dims(self):
		return self.data_dim

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def get_metadata(self):
		return {
			'n': self.len,
			'data_dim': self.data_dim,
			}    

class CustomFCEdatasetSegments(Dataset):
	def __init__(self, X, Y, U, device='cpu'):
		self.device = device
		self.x = torch.from_numpy(X).to(device)
		self.y = torch.from_numpy(Y).to(device)
		self.u = torch.from_numpy(U).to(device)
		self.len = self.x.shape[0]
		self.data_dim = self.x.shape[1]
		#print('data loaded on {}'.format(self.x.device))

	def get_dims(self):
		return self.data_dim

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.x[index], self.y[index], self.u[index]



def to_one_hot(x, m=None):
	"batch one hot"
	if type(x) is not list:
		x = [x]
	if m is None:
		ml = []
		for xi in x:
			ml += [xi.max() + 1]
		m = max(ml)
	dtp = x[0].dtype
	xoh = []
	for i, xi in enumerate(x):
		xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
		xoh[i][np.arange(xi.size), xi.astype(np.int)] = 1
	return xoh

# ------------------------
# define ebm FCE object
# ------------------------

class ebmFCE( object ):
	"""
	train an energy based model using noise contrastive estimation
	"""

	def __init__( self, data, energy_MLP, flow_model ):
		self.data = data
		self.energy_MLP = energy_MLP
		self.ebm_norm   = -5.
		self.flow_model = flow_model # flow model, must have sample and log density capabilities
		self.noise_samples = None
		self.device = 'gpu' if torch.cuda.is_available() else 'cpu'

	def sample_noise( self, n ):
		return self.flow_model.sample( n )[-1].detach().numpy() 

	def noise_logpdf( self, dat):
		"""
		compute log density under flow model
		"""
		zs, prior_logprob, log_det = self.flow_model( dat )
		flow_logdensity = ( prior_logprob + log_det )
		return flow_logdensity


	def train_ebm_fce( self, epochs=500, lr=.0001 ):
		"""
		FCE training of EBM model
		"""
		print('Training energy based model using FCE')

		# sample noise data
		n = self.data.shape[0]
		self.noise_samples = self.sample_noise( n ) #self.noise_dist.sample( n )
		# define classification labels
		y = np.array( [0] * n + [1] * n ) 

		# define 
		dat_fce = CustomFCEdataset( np.vstack(( self.data, self.noise_samples)).astype(np.float32), to_one_hot(y)[0].astype(np.float32), device=self.device )
		fce_loader = DataLoader( dat_fce, shuffle=True, batch_size=128 )

		# define log normalization constant
		ebm_norm = self.ebm_norm #-5.
		logNorm = torch.from_numpy( np.array(ebm_norm).astype(np.float32)) #, device=dat_fce.device, requires_grad=True ) 
		logNorm.requires_grad_()

		# define optimizer 
		optimizer = optim.Adam( list(self.energy_MLP.parameters()) + [logNorm],  lr=.0001 )
		self.energy_MLP.train()

		# begin optimization
		loss_criterion = nn.BCELoss()

		for e in range(epochs):
			num_correct = 0
			loss_val = 0
			for _, (dat, label) in enumerate( fce_loader ):
				# noise model probs:
				noise_logpdf = self.noise_logpdf( dat ).view(-1,1) #torch.tensor( self.noise_dist.logpdf( dat ).astype(np.float32) ).view(-1,1)

				# get ebm model probs:
				ebm_logpdf = ( self.energy_MLP(dat) + logNorm )

				# define logits
				logits = torch.cat( (ebm_logpdf - noise_logpdf, noise_logpdf - ebm_logpdf), 1)

				# compute accuracy:
				num_correct += ( logits.argmax(1) == label.argmax(1)).sum().item()

				# define loss
				loss = loss_criterion( torch.sigmoid(logits), label )
				loss_val += loss.item()

				# take gradient step
				self.energy_MLP.zero_grad()

				optimizer.zero_grad()

				# compute gradients
				loss.backward()

				# update parameters
				optimizer.step()

			# print some statistics
			print('epoch {} \tloss: {}\taccuracy: {}'.format(e, np.round(loss_val,4), np.round(num_correct/(2*n), 3)))

		self.ebm_norm = logNorm.item()


	def reset_noise( self ):
		self.noise_samples = self.sample_noise( self.noise_samples.shape[0] ) 

	def train_flow_fce( self, epochs=50, lr=1e-4, objConstant = -1.0 ):
		"""
		FCE training of EBM model
		"""
		print('Training flow contrastive noise for FCE')

		# noise data already sampled during EBM training
		n = self.data.shape[0]
		self.reset_noise() 
		# define classification labels
		y = np.array( [0] * n + [1] * n ) 

		# define 
		dat_fce = CustomFCEdataset( np.vstack(( self.data, self.noise_samples)).astype(np.float32), to_one_hot(y)[0].astype(np.float32), device=self.device )
		fce_loader = DataLoader( dat_fce, shuffle=True, batch_size=128 )


		# define optimizer 
		optimizer = optim.Adam(self.flow_model.parameters(), lr=lr, weight_decay=1e-5) # todo tune WD
		self.flow_model.train()

		# begin optimization
		loss_criterion = nn.BCELoss()

		for e in range(epochs):
			num_correct = 0
			loss_val = 0
			for _, (dat, label) in enumerate( fce_loader ):
				# noise model probs:
				noise_logpdf = self.noise_logpdf( dat ).view(-1,1) #torch.tensor( self.noise_dist.logpdf( dat ).astype(np.float32) ).view(-1,1)

				# get ebm model probs:
				ebm_logpdf = ( self.energy_MLP(dat) + self.ebm_norm )

				# define logits
				logits = torch.cat( (ebm_logpdf - noise_logpdf, noise_logpdf - ebm_logpdf), 1)
				logits *= objConstant

				# compute accuracy:
				num_correct += ( logits.argmax(1) == label.argmax(1)).sum().item()

				# define loss
				loss = loss_criterion( torch.sigmoid(logits), label ) 
				loss_val += loss.item()

				# take gradient step
				self.flow_model.zero_grad()

				optimizer.zero_grad()

				# compute gradients
				loss.backward()

				# update parameters
				optimizer.step()

			# print some statistics
			print('epoch {} \tloss: {}\taccuracy: {}'.format(e, np.round(loss_val,4), np.round(1-num_correct/(2*n), 3)))



class ebmFCEsegments( object ):
	"""
	train an energy based model using noise contrastive estimation
	where we assume we observe data from multiple segments/classes 
	this is useful for nonlinear ICA and semi supervised learning !
	"""

	def __init__( self, data, segments, energy_MLP, flow_model, verbose=False ):
		self.data = data
		self.segments = segments
		self.contrastSegments = ( np.ones( self.segments.shape ) / self.segments.shape[1] ).astype(np.float32)
		self.energy_MLP = energy_MLP
		self.ebm_norm   = -5.
		self.hidden_dim = self.energy_MLP.linearLast.weight.shape[0]
		self.n_segments = self.segments.shape[1]
		self.ebm_finalLayer = torch.tensor( np.ones(( self.hidden_dim, self.n_segments )).astype(np.float32) )		
		#self.ebm_finalLayer = torch.tensor( np.random.random(( self.hidden_dim, self.n_segments )).astype(np.float32) )
		self.flow_model = flow_model # flow model, must have sample and log density capabilities
		self.noise_samples = None
		self.device  = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.verbose = verbose

	def sample_noise( self, n ):
		if self.device == 'cuda':
			return self.flow_model.module.sample( n )[-1].detach().cpu().numpy() 
		else:
			return self.flow_model.sample( n )[-1].detach().numpy() 

	def noise_logpdf( self, dat):
		"""
		compute log density under flow model
		"""
		zs, prior_logprob, log_det = self.flow_model( dat )
		flow_logdensity = ( prior_logprob + log_det )
		return flow_logdensity

	def compute_ebm_logpdf( self, dat, seg, logNorm, augment=False ):
		act_allLayer = torch.mm( self.energy_MLP(dat), self.ebm_finalLayer )
		
		if augment:
			# we augment the feature extractor 
			act_allLayer += torch.mm( self.energy_MLP(dat) * self.energy_MLP(dat), self.ebm_finalLayer * self.ebm_finalLayer )

		# now select relevant layers by multiplying by mask matrix and reducing (and adding log norm)
		act_segment = (act_allLayer * seg).sum(1) + logNorm 

		return act_segment


	def train_ebm_fce( self, epochs=500, lr=.0001, cutoff=None, augment=False, finalLayerOnly=False, useVAT=False ):
		"""
		FCE training of EBM model
		"""
		if self.verbose:
			print('Training energy based model using FCE' + useVAT * ' with VAT penalty')

		if cutoff is None:
			cutoff = 1.00 # will basically only stop with perfect classification

		# sample noise data
		n = self.data.shape[0]
		self.noise_samples = self.sample_noise( n ) #self.noise_dist.sample( n )
		# define classification labels
		y = np.array( [0] * n + [1] * n ) 

		# define 
		dat_fce = CustomFCEdatasetSegments( np.vstack(( self.data, self.noise_samples)).astype(np.float32), to_one_hot(y)[0].astype(np.float32), np.vstack( (self.segments, self.contrastSegments ) ), device=self.device  )
		fce_loader = DataLoader( dat_fce, shuffle=True, batch_size=128 )

		# define log normalization constant
		ebm_norm = self.ebm_norm #-5.
		logNorm = torch.from_numpy( np.array(ebm_norm).astype(np.float32)).float().to( self.device ) #, device=dat_fce.device, requires_grad=True ) 
		logNorm.requires_grad_()

		# 
		self.ebm_finalLayer.requires_grad_()

		# define optimizer 
		if finalLayerOnly:
			# only train the final layer, this is the equivalent of g(y) in 
			# IMCA manuscript.
			optimizer = optim.Adam( [self.ebm_finalLayer] +[logNorm],  lr=lr )	
		else:
			optimizer = optim.Adam( list(self.energy_MLP.parameters()) + [self.ebm_finalLayer] +[logNorm],  lr=lr )
		
		self.energy_MLP.to( self.device )
		self.energy_MLP.train()

		# begin optimization
		loss_criterion = nn.BCELoss()

		use_cuda = torch.cuda.is_available()
		if use_cuda:
			self.energy_MLP.cuda()
			self.energy_MLP = torch.nn.DataParallel(self.energy_MLP, device_ids=range(torch.cuda.device_count()))
			cudnn.benchmark = True

		for e in range(epochs):
			num_correct = 0
			loss_val = 0
			for _, (dat, label, seg) in enumerate( fce_loader ):
				# consider adding VAT loss
				if useVAT:
					vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
					lds = vat_loss( self.energy_MLP, dat )

				# noise model probs:
				noise_logpdf = self.noise_logpdf( dat ).view(-1,1) #torch.tensor( self.noise_dist.logpdf( dat ).astype(np.float32) ).view(-1,1)

				# pass to correct device:
				if use_cuda:
					dat = dat.to( self.device )
					seg = seg.to( self.device )
					label = label.to( self.device )
					#dat, seg = dat.cuda(), seg.cuda()

				# get ebm log pdf
				ebm_logpdf = self.compute_ebm_logpdf( dat, seg, logNorm, augment=augment ).view(-1,1)

				# define logits
				logits = torch.cat( (ebm_logpdf - noise_logpdf, noise_logpdf - ebm_logpdf), 1)
				logits.to( self.device )

				# compute accuracy:
				num_correct += ( logits.argmax(1) == label.argmax(1)).sum().item()

				# define loss
				loss = loss_criterion( torch.sigmoid(logits), label )
				if useVAT:
					loss += 1 * lds 

				loss_val += loss.item()

				# take gradient step
				self.energy_MLP.zero_grad()

				optimizer.zero_grad()

				# compute gradients
				loss.backward()

				# update parameters
				optimizer.step()

			# print some statistics
			if self.verbose:
				print('epoch {} \tloss: {}\taccuracy: {}'.format(e, np.round(loss_val,4), np.round(num_correct/(2*n), 3)))
			if num_correct/(2*n) > cutoff:
				# stop training
				if self.verbose:
					print('epoch {}\taccuracy: {}'.format(e, np.round(num_correct/(2*n), 3)))
					print('cutoff value satisfied .. stopping training\n----------\n')
				break 

		self.ebm_norm = logNorm.item()


	def reset_noise( self ):
		self.noise_samples = self.sample_noise( self.noise_samples.shape[0] ) 


	def pretrain_flow_model( self, epochs=50, lr=1e-4 ):
		"""
		pertraining of flow model using MLE
		"""
		optimizer = optim.Adam(self.flow_model.parameters(), lr=1e-4, weight_decay=1e-5) # todo tune WD
		#print("number of params: ", sum(p.numel() for p in model_flow.parameters()))

		dset = CustomSyntheticDatasetDensity(self.data.astype(np.float32), device=self.device ) 
		train_loader = DataLoader( dset, shuffle=True, batch_size=128 )

		# run optimization
		loss_vals = []

		use_cuda = torch.cuda.is_available()
		if use_cuda:
			self.flow_model.to( self.device )
			self.flow_model = torch.nn.DataParallel(self.flow_model, device_ids=range(torch.cuda.device_count()))
			#self.flow_model.to( self.device )
			cudnn.benchmark = True
			print("using gpus! " + str(self.device) )

		self.flow_model.train()
		for e in range( epochs ):
			loss_val = 0
			for _, dat in enumerate( train_loader ):
				if use_cuda:
					dat = dat.cuda()
					dat = Variable( dat )
				zs, prior_logprob, log_det = self.flow_model( dat )
				logprob = prior_logprob + log_det
				loss = - torch.sum(logprob) # NLL

				#print(loss.item())
				loss_val += loss.item()

				# 
				self.flow_model.zero_grad()
				optimizer.zero_grad()

				# compute gradients
				loss.backward()

				# update parameters
				optimizer.step()

			if self.verbose:
				print('epoch {}/{} \tloss: {}'.format(e, epochs, loss_val))
			loss_vals.append( loss_val )


	def train_flow_fce( self, epochs=50, lr=1e-4, objConstant = -1.0, cutoff=None ):
		"""
		FCE training of EBM model
		"""
		if self.verbose:
			print('Training flow contrastive noise for FCE')

		if cutoff is None:
			cutoff = 0. # basically only stop for perfect misclassification

		# noise data already sampled during EBM training
		n = self.data.shape[0]
		self.reset_noise() 
		# define classification labels
		y = np.array( [0] * n + [1] * n ) 

		# define 
		dat_fce = CustomFCEdatasetSegments( np.vstack(( self.data, self.noise_samples)).astype(np.float32), to_one_hot(y)[0].astype(np.float32), np.vstack( (self.segments, self.segments ) ), device=self.device )
		fce_loader = DataLoader( dat_fce, shuffle=True, batch_size=128 )

		# define optimizer 
		optimizer = optim.Adam(self.flow_model.parameters(), lr=lr, weight_decay=1e-5) # todo tune WD
		
		use_cuda = torch.cuda.is_available()
		self.flow_model.to( self.device )
		self.flow_model.train()

		# begin optimization
		loss_criterion = nn.BCELoss()

		for e in range(epochs):
			num_correct = 0
			loss_val = 0
			for _, (dat, label, seg) in enumerate( fce_loader ):
				# pass to correct device:
				if use_cuda:
					dat = dat.to( self.device )
					seg = seg.to( self.device )
					label = label.to( self.device )

				# noise model probs:
				noise_logpdf = self.noise_logpdf( dat ).view(-1,1) #torch.tensor( self.noise_dist.logpdf( dat ).astype(np.float32) ).view(-1,1)

				# get ebm model probs:
				ebm_logpdf = self.compute_ebm_logpdf( dat, seg, self.ebm_norm ).view(-1,1)

				# define logits
				logits = torch.cat( (ebm_logpdf - noise_logpdf, noise_logpdf - ebm_logpdf), 1)
				logits *= objConstant

				# compute accuracy:
				num_correct += ( logits.argmax(1) == label.argmax(1)).sum().item()

				# define loss
				loss = loss_criterion( torch.sigmoid(logits), label ) 
				loss_val += loss.item()

				# take gradient step
				self.flow_model.zero_grad()

				optimizer.zero_grad()

				# compute gradients
				loss.backward()

				# update parameters
				optimizer.step()

			# print some statistics
			if self.verbose:
				print('epoch {} \tloss: {}\taccuracy: {}'.format(e, np.round(loss_val,4), np.round(1-num_correct/(2*n), 3)))
			if 1-num_correct/(2*n) < cutoff:
				if self.verbose:
					print('epoch {}\taccuracy: {}'.format(e, np.round(1-num_correct/(2*n), 3)))	
					print('cutoff value satisfied .. stopping training\n----------\n')
				break

	def unmixSamples( self, data, modelChoice ):
		"""
		perform unmixing of samples 
		"""
		if modelChoice == 'EBM':
			# unmix using EBM:
			if self.device == 'gpu':
				recov = self.energy_MLP( torch.tensor( data.astype(np.float32) ) ).detach().numpy()
			else:
				recov = self.energy_MLP( torch.tensor( data.astype(np.float32) ) ).detach().cpu().numpy()				
		else:
			# unmix using flow model
			if self.device == 'cpu':
				recov = self.flow_model( torch.tensor( data.astype(np.float32) ) )[0][-1].detach().numpy()
			else:
				recov = self.flow_model( torch.tensor( data.astype(np.float32) ) )[0][-1].detach().cpu().numpy()

		return recov 
