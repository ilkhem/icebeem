### Generate data based on the non-stationary, non-linear ICA model of Hyvarinen & Morioka (2016)
#
# data is generated using two distinct non-linearity functions: leaky-ReLU and sigmoid functions 
#
#

import numpy as np
from sklearn.preprocessing import scale
from scipy.stats import ortho_group

def leaky_ReLU_1d( d, negSlope ):
  """
  one dimensional implementation of leaky ReLU
  """
  if d > 0:
    return d
  else:
    return d * negSlope

leaky1d = np.vectorize( leaky_ReLU_1d )

def leaky_ReLU( D, negSlope ):
  """
  implementation of leaky ReLU activation function
  """
  assert negSlope > 0 # must be positive 
  return leaky1d( D, negSlope)

def sigmoidAct( x ):
  """
  one dimensional application of sigmoid activation function
  """
  return 1./( 1 + np.exp( -1 * x ) )

def generateUniformMat( Ncomp, condT ):
  """  
  generate a random matrix by sampling each element uniformly at random 
  check condition number versus a condition threshold
  """
  A = np.random.uniform(0,2, (Ncomp, Ncomp)) - 1
  for i in range(Ncomp):
    A[:,i] /= np.sqrt( (A[:,i]**2).sum())
  
  while np.linalg.cond(A) > condT:
    # generate a new A matrix!
    A = np.random.uniform(0,2, (Ncomp, Ncomp)) - 1 
    for i in range(Ncomp):
      A[:,i] /= np.sqrt( (A[:,i]**2).sum())
      
  return A

def generateUniformMat_minMax( Ncomp, condT, minVal=.5, maxVal=1.5 ):
  """  
  generate a random matrix by sampling each element uniformly at random 
  check condition number versus a condition threshold
  """
  A = np.random.uniform(minVal, maxVal, (Ncomp, Ncomp)) 
  for i in range(Ncomp):
    A[:,i] /= np.sqrt( (A[:,i]**2).sum())
  
  while np.linalg.cond(A) > condT:
    # generate a new A matrix!
    A = np.random.uniform(minVal, maxVal, (Ncomp, Ncomp))  
    for i in range(Ncomp):
      A[:,i] /= np.sqrt( (A[:,i]**2).sum())
      
  return A




def genTCLdata( Ncomp, Nlayer, Nsegment, NsegmentObs, source='Laplace', NonLin='leaky',  negSlope=.2, Niter4condThresh =1e4 ):
  """
  
  generate multivariate data based on the non-stationary non-linear ICA model of Hyvarinen & Morioka (2016)

  INPUT
      - Ncomp: number of components (i.e., dimensionality of the data)
      - Nlayer: number of non-linear layers!
      - Nsegment: number of data segments to generate
      - NsegmentObs: number of observations per segment 
      - source: either Laplace or Gaussian, denoting distribution for latent sources
      - NonLin: linearity employed in non-linear mixing. Can be one of "leaky" = leakyReLU or "sigmoid"=sigmoid       
        Specifically for leaky activation we also have:
        	- negSlope: slope for x < 0 in leaky ReLU
	        - Niter4condThresh: number of random matricies to generate to ensure well conditioned 
  OUTPUT:
    - output is a dictionary with the following values: 
  	  - sources: original non-stationary source
	    - obs: mixed sources 
	    - labels: segment labels (indicating the non stationarity in the data)
  
  
  """
  
  # check input is correct
  assert NonLin in ['leaky', 'sigmoid']

  # generate non-stationary data:
  Nobs = NsegmentObs * Nsegment # total number of observations
  labels = np.array( [0] * Nobs ) # labels for each observation (populate below)
  
  # generate data, which we will then modulate in a non-stationary manner:
  if source=='Laplace':
    dat = np.random.laplace( 0, 1, (Nobs, Ncomp) )
    dat = scale( dat ) # set to zero mean and unit variance 
  elif source=='Gaussian':
    dat = np.random.normal( 0, 1, (Nobs, Ncomp) )
    dat = scale( dat )
  else:
    raise Exception("wrong source distribution")

  
  # get modulation parameters
  modMat = np.random.uniform( 0 , 1, (Ncomp, Nsegment) )
  
  # now we adjust the variance within each segment in a non-stationary manner
  for seg in range(Nsegment):
    segID = range( NsegmentObs*seg, NsegmentObs*(seg+1) )
    dat[ segID, :] = np.multiply( dat[ segID, :], modMat[:, seg])
    labels[ segID] = seg 
    
  # now we are ready to apply the non-linear mixtures: 
  mixedDat = np.copy(dat)

  # generate mixing matrices:  
  # will generate random uniform matrices and check their condition number based on following simulations:
  condList = []
  for i in range(int(Niter4condThresh)):
    # A = np.random.uniform(0,1, (Ncomp, Ncomp))
    A = np.random.uniform(1,2, (Ncomp, Ncomp)) #- 1
    for i in range(Ncomp):
      A[:,i] /= np.sqrt( (A[:,i]**2).sum())
    condList.append( np.linalg.cond( A ))

  condThresh = np.percentile( condList, 15 ) # only accept those below 25% percentile 

  # now we apply layers of non-linearity (just one for now!). Note the order will depend on natural of nonlinearity! (either additive or more general!)  
  mixingList = []
  for l in range(Nlayer-1):
    # generate causal matrix first:
    A = generateUniformMat( Ncomp, condThresh )
    mixingList.append(A)

    # we first apply non-linear function, then causal matrix!
    if NonLin == 'leaky':
      mixedDat = leaky_ReLU( mixedDat, negSlope )
    elif NonLin == 'sigmoid':
      mixedDat = sigmoidAct( mixedDat )
    # apply mixing:
    mixedDat = np.dot( mixedDat, A )

  return {'source': dat, 'obs': mixedDat, 'labels': labels, 'mixing': mixingList, 'var':modMat }



def genTCLdataOrtho( Ncomp, Nlayer, Nsegment, NsegmentObs, source='Laplace', NonLin='leaky',  negSlope=.2, Niter4condThresh =1e4 ):
  """
  
  generate multivariate data based on the non-stationary non-linear ICA model of Hyvarinen & Morioka (2016)

  we generate mixing matrices using random orthonormal matrices

  INPUT
      - Ncomp: number of components (i.e., dimensionality of the data)
      - Nlayer: number of non-linear layers!
      - Nsegment: number of data segments to generate
      - NsegmentObs: number of observations per segment 
      - source: either Laplace or Gaussian, denoting distribution for latent sources
      - NonLin: linearity employed in non-linear mixing. Can be one of "leaky" = leakyReLU or "sigmoid"=sigmoid       
        Specifically for leaky activation we also have:
          - negSlope: slope for x < 0 in leaky ReLU
          - Niter4condThresh: number of random matricies to generate to ensure well conditioned 
  OUTPUT:
    - output is a dictionary with the following values: 
      - sources: original non-stationary source
      - obs: mixed sources 
      - labels: segment labels (indicating the non stationarity in the data)
  
  
  """
  
  # check input is correct
  assert NonLin in ['leaky', 'sigmoid']

  # generate non-stationary data:
  Nobs = NsegmentObs * Nsegment # total number of observations
  labels = np.array( [0] * Nobs ) # labels for each observation (populate below)
  
  # generate data, which we will then modulate in a non-stationary manner:
  if source=='Laplace':
    dat = np.random.laplace( 0, 1, (Nobs, Ncomp) )
    dat = scale( dat ) # set to zero mean and unit variance 
  elif source=='Gaussian':
    dat = np.random.normal( 0, 1, (Nobs, Ncomp) )
    dat = scale( dat )
  else:
    raise Exception("wrong source distribution")

  
  # get modulation parameters
  modMat = np.random.uniform( 0.01 , 3, (Ncomp, Nsegment) )

  if False:
    meanMat = np.random.uniform( 0, 5, (Ncomp, Nsegment) )
  else:
    meanMat = np.zeros( (Ncomp, Nsegment))
  # now we adjust the variance within each segment in a non-stationary manner
  for seg in range(Nsegment):
    segID = range( NsegmentObs*seg, NsegmentObs*(seg+1) )
    dat[ segID, :] = np.multiply( dat[ segID, :], modMat[:, seg] )
    dat[ segID, :] = np.add( dat[ segID, :], meanMat[:, seg] )
    labels[ segID] = seg 
    
  # now we are ready to apply the non-linear mixtures: 
  mixedDat = np.copy(dat)

  # generate mixing matrices:  
  # now we apply layers of non-linearity (just one for now!). Note the order will depend on natural of nonlinearity! (either additive or more general!)  
  mixingList = []
  for l in range(Nlayer-1):
    # generate causal matrix first:
    A = ortho_group.rvs( Ncomp ) #generateUniformMat( Ncomp, condThresh )
    mixingList.append(A)

    # we first apply non-linear function, then causal matrix!
    if NonLin == 'leaky':
      mixedDat = leaky_ReLU( mixedDat, negSlope )
    elif NonLin == 'sigmoid':
      mixedDat = sigmoidAct( mixedDat )
    # apply mixing:
    mixedDat = np.dot( mixedDat, A )

  return {'source': dat, 'obs': mixedDat, 'labels': labels, 'mixing': mixingList, 'var':modMat }





