# THIS IS A LEGACY CODE, SAFE TO REMOVE

### implementation of MLP to be used for DEEN density estimator
#
#
#

import torch
import argparse

import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

from sklearn.metrics import pairwise_distances

class MLPlayer( nn.Module ):
  """
  implement basic module for MLP 
  
  note that this module keeps the dimensions fixed! will implement a mapping from a 
  vector of dimension input_size to another vector of dimension input_size
  
  """
  
  def __init__( self, input_size, output_size = None, activation_function = nn.functional.relu, use_bn=False ):
    super( MLPlayer, self).__init__()
    if output_size is None:
      output_size = input_size
    self.activation_function = activation_function
    self.linear_layer = nn.Linear( input_size, output_size )
    self.use_bn = use_bn
    self.bn_layer = nn.BatchNorm1d( input_size )
    
  def forward( self, x ):
    if self.use_bn:
      x = self.bn_layer( x )
    linear_act = self.linear_layer( x )
    H_x = self.activation_function( linear_act )
    return H_x


class MLP( nn.Module ):
  """
  define a MLP network!
  """
  def __init__( self, input_size, hidden_size, n_layers, activation_function = F.relu ):
    """
    
    Input:
     - input_size  : dimension of input data (e.g., 784 for MNIST)
     - hidden_size : size of hidden representations 
     - n_layers    : number of hidden layers 
     
    """
    super( MLP, self ).__init__()
    
    output_size = 1 # because we approximating a log density, output should be scalar!

    self.activation_function = activation_function
    self.linear1st = nn.Linear( input_size, hidden_size ) # map from data dim to dimension of hidden units
    self.Layers = nn.ModuleList( [MLPlayer( hidden_size, activation_function=self.activation_function ) for _ in range(n_layers) ] )
    self.linearLast = nn.Linear( hidden_size, output_size ) # map from dimension of hidden units to dimension of output
    
  def forward( self, x ):
    """
    forward pass through resnet
    """
    x = self.linear1st( x )
    for current_layer in self.Layers :
      x = current_layer( x )
    x =  self.linearLast( x ) 
    return x


# comment




class MLP_general( nn.Module ):
  """
  define a MLP network - this is a more general class than MLP above, allows for user to specify
  the dimensions at each layer of the network
  """
  def __init__( self, input_size,  hidden_size, n_layers, output_size=None, activation_function = F.relu, use_bn=False ):
    """
    
    Input:
     - input_size  : dimension of input data (e.g., 784 for MNIST)
     - hidden_size : list of hidden representations, one entry per layer 
     - n_layers    : number of hidden layers 
     
    """
    super( MLP_general, self ).__init__()
    
    if output_size is None:
      output_size = 1 # because we approximating a log density, output should be scalar!

    self.use_bn = use_bn
    self.activation_function = activation_function
    self.linear1st = nn.Linear( input_size, hidden_size[0] ) # map from data dim to dimension of hidden units
    self.Layers = nn.ModuleList( [MLPlayer( hidden_size[i-1], hidden_size[i], activation_function=self.activation_function, use_bn=self.use_bn ) for i in range(1,n_layers) ] )
    self.linearLast = nn.Linear( hidden_size[-1], output_size ) # map from dimension of hidden units to dimension of output
    
  def forward( self, x ):
    """
    forward pass through resnet
    """
    x = self.linear1st( x )
    for current_layer in self.Layers :
      x = current_layer( x )
    x =  self.linearLast( x ) 
    return x


class smoothReLU( nn.Module ):
  """
  smooth ReLU activation function
  """

  def __init__( self, beta=1 ):
    super(smoothReLU, self).__init__()
    self.beta = 1

  def forward(self, x):
    return x / (1 + torch.exp(-self.beta * x ) )

def compute_sigma_saremi( dat ):
  """
  compute median, mean and max distance between two samples as 
  suggested in Saremi & Hyvarinen (2019)
  """

  if dat.shape[0] > 5000:
    dat = dat[:5000,:] # otherwise computing pairwise dist takes too long!

  D = pairwise_distances( dat )[ np.triu_indices( dat.shape[0] ) ] / (2 * np.sqrt( dat.shape[1] ))

  return {'meanD':np.mean( D ), 
  'p15D':np.percentile( D, 15 ),
  'p05D':np.percentile( D, 5 ),
  'p25D':np.percentile( D, 25 ),
  'p50D':np.percentile( D, 50 ), 'maxD':np.max( D )}


class CustomSyntheticDataset(Dataset):
    def __init__(self, X, Xtilde, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(device)
        self.xtilde = torch.from_numpy(Xtilde).to(device)
        self.len = self.x.shape[0]
        #self.aux_dim = self.xtilde.shape[1]
        #self.data_dim = self.x.shape[1]
        #self.nps = int(self.len / self.aux_dim)
        print('data loaded on {}'.format(self.x.device))

    def get_dims(self):
        return self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.xtilde[index]

    def get_metadata(self):
        return {
                #'nps': self.nps,
                #'ns': self.aux_dim,
                'n': self.len,
                'data_dim': self.data_dim,
                #'aux_dim': self.aux_dim,
                }    