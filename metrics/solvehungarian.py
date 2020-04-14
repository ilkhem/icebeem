import numpy as np 
from scipy.optimize import linear_sum_assignment

def SolveHungarian( recov, source ):
	"""
	compute maximum correlations between true indep components and estimated components 
	"""
	Ncomp = source.shape[1]
	CorMat = (np.abs(np.corrcoef( recov.T, source.T ) ) )[:Ncomp, Ncomp:]
	ii = linear_sum_assignment( -1*CorMat )

	return CorMat[ii].mean(), CorMat, ii