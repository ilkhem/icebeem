import numpy as np
import torch
from scipy.stats import ortho_group
from scipy.stats import random_correlation
from sklearn.preprocessing import scale
from torch.utils.data import Dataset

from .utils import to_one_hot


class ConditionalDataset(Dataset):
    """
    a Dataset object holding a tuple (x,y): observed and auxiliary variable
    used in `models.ivae.ivae_wrapper.IVAE_wrapper()`
    """

    def __init__(self, X, Y, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X)  # .to(device)
        self.y = torch.from_numpy(Y)  # .to(device)  # if discrete, then this should be one_hot
        self.len = self.x.shape[0]
        self.aux_dim = self.y.shape[1]
        self.data_dim = self.x.shape[1]
        self.latent_dim = self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def get_dims(self):
        return self.data_dim, self.latent_dim, self.aux_dim


class SimpleDataset(Dataset):
    """
    a Dataset object holding a single observed variable x
    used in `models.icebeem_FCE.ebmFCEsegments()`
    """

    def __init__(self, X, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(device)
        self.len = self.x.shape[0]
        self.data_dim = self.x.shape[1]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index]


class ContrastiveSimpleDataset(Dataset):
    """
    a Dataset object holding a pair (x, label): x is observed variable which is either from our model or a
    contrastive dist (usually a flow) and lable is in {0,1}
    used in `models.icebeem_FCE.ebmFCE()`
    """

    def __init__(self, X, Y, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(device)
        self.y = torch.from_numpy(Y).to(device)
        self.len = self.x.shape[0]
        self.data_dim = self.x.shape[1]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class ContrastiveConditionalDataset(Dataset):
    """
    a Dataset object holding a tuple (x, y, label): x is observed variable which is either from our model or a
    contrastive dist (usually a flow), y is the auxiliary variable and label is in {0,1}
    used in `models.icebeem_FCE.ebmFCEsegments()`
    """

    def __init__(self, X, Y, U, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(device)
        self.y = torch.from_numpy(Y).to(device)
        self.u = torch.from_numpy(U).to(device)
        self.len = self.x.shape[0]
        self.data_dim = self.x.shape[1]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.u[index]


def leaky_ReLU_1d(d, negSlope):
    """
    one dimensional implementation of leaky ReLU
    """
    if d > 0:
        return d
    else:
        return d * negSlope


leaky1d = np.vectorize(leaky_ReLU_1d)


def leaky_ReLU(D, negSlope):
    """
    implementation of leaky ReLU activation function
    """
    assert negSlope > 0  # must be positive
    return leaky1d(D, negSlope)


def sigmoidAct(x):
    """
    one dimensional application of sigmoid activation function
    """
    return 1. / (1 + np.exp(-1 * x))


def generateUniformMat(Ncomp, condT):
    """
    generate a random matrix by sampling each element uniformly at random
    check condition number versus a condition threshold
    """
    A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
    for i in range(Ncomp):
        A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    while np.linalg.cond(A) > condT:
        # generate a new A matrix!
        A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
        for i in range(Ncomp):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    return A


def generateUniformMat_minMax(Ncomp, condT, minVal=.5, maxVal=1.5):
    """
    generate a random matrix by sampling each element uniformly at random
    check condition number versus a condition threshold
    """
    A = np.random.uniform(minVal, maxVal, (Ncomp, Ncomp))
    for i in range(Ncomp):
        A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    while np.linalg.cond(A) > condT:
        # generate a new A matrix!
        A = np.random.uniform(minVal, maxVal, (Ncomp, Ncomp))
        for i in range(Ncomp):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    return A


def gen_nonstationary_data(Ncomp, Nlayer, Nsegment, NsegmentObs, source='Laplace', NonLin='leaky', negSlope=.2,
                           Niter4condThresh=1e4, seed=1):
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
    np.random.seed(seed)
    # check input is correct
    assert NonLin in ['leaky', 'sigmoid']

    # generate non-stationary data:
    Nobs = NsegmentObs * Nsegment  # total number of observations
    labels = np.array([0] * Nobs)  # labels for each observation (populate below)

    # generate data, which we will then modulate in a non-stationary manner:
    if source == 'Laplace':
        dat = np.random.laplace(0, 1, (Nobs, Ncomp))
        dat = scale(dat)  # set to zero mean and unit variance
    elif source == 'Gaussian':
        dat = np.random.normal(0, 1, (Nobs, Ncomp))
        dat = scale(dat)
    else:
        raise Exception("wrong source distribution")

    # get modulation parameters
    modMat = np.random.uniform(0, 1, (Ncomp, Nsegment))

    # now we adjust the variance within each segment in a non-stationary manner
    for seg in range(Nsegment):
        segID = range(NsegmentObs * seg, NsegmentObs * (seg + 1))
        dat[segID, :] = np.multiply(dat[segID, :], modMat[:, seg])
        labels[segID] = seg

    # now we are ready to apply the non-linear mixtures:
    mixedDat = np.copy(dat)

    # generate mixing matrices:
    # will generate random uniform matrices and check their condition number based on following simulations:
    condList = []
    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (Ncomp, Ncomp))  # - 1
        for i in range(Ncomp):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 15)  # only accept those below 25% percentile

    # now we apply layers of non-linearity (just one for now!). Note the order will depend on natural of nonlinearity!
    # (either additive or more general!)
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = generateUniformMat(Ncomp, condThresh)
        mixingList.append(A)

        # we first apply non-linear function, then causal matrix!
        if NonLin == 'leaky':
            mixedDat = leaky_ReLU(mixedDat, negSlope)
        elif NonLin == 'sigmoid':
            mixedDat = sigmoidAct(mixedDat)
        # apply mixing:
        mixedDat = np.dot(mixedDat, A)

    return {'source': dat, 'obs': mixedDat, 'labels': labels, 'mixing': mixingList, 'var': modMat}


def gen_IMCA_data(Ncomp, Nlayer, Nsegment, NsegmentObs, BaseCovariance, NonLin='leaky', negSlope=.2,
                  Niter4condThresh=1e4, seed=1, varyMean=False):
    """
    generate data from an IMCA model where latent sources follow a
    MoG distribution conditional on each segment
    Crucial difference is latents are NOT conditionally
    independent (unless we specify that BaseCovariance is diagonal!)

    INPUT
        - Ncomp: number of components (i.e., dimensionality of the data)
        - Nlayer: number of non-linear layers!
        - Nsegment: number of data segments to generate
        - NsegmentObs: number of observations per segment
        - source: either Laplace or Gaussian, denoting distribution for latent sources
        - BaseCovariance: base measure covariance (make diagonal for cond. indep latents)
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
    np.random.seed(seed)
    # define some params
    Nobs = NsegmentObs * Nsegment  # total number of observations
    labels = np.array([0] * Nobs)  # labels for each observation (populate below)
    latents = np.zeros((Nobs, Ncomp))

    # generate segment variance for latent variables
    modMat = np.random.uniform(0.01, 3, (Ncomp, Nsegment)) ** 2  # this is to make consistent with TCL experiments !

    if varyMean:
        meanMat = np.random.uniform(-3, 3, (Ncomp, Nsegment))
    else:
        meanMat = np.zeros((Ncomp, Nsegment))

    for i in range(Nsegment):
        # define presicion for segment i
        Pres_i = np.linalg.inv(BaseCovariance) * (.5 / Ncomp) + np.diag(1 / modMat[:, i])
        latents[(i * NsegmentObs):((i + 1) * NsegmentObs), :] = np.random.multivariate_normal(mean=np.zeros((Ncomp,)),
                                                                                              cov=np.linalg.inv(Pres_i),
                                                                                              size=NsegmentObs)
        latents[(i * NsegmentObs):((i + 1) * NsegmentObs), :] = np.add(
            latents[(i * NsegmentObs):((i + 1) * NsegmentObs), :], meanMat[:, i])
        labels[(i * NsegmentObs):((i + 1) * NsegmentObs)] = i

    # now we are ready to apply the non-linear mixtures:
    mixedDat = np.copy(latents)

    # generate mixing matrices:
    # now we apply layers of non-linearity (just one for now!). Note the order will depend on natural of nonlinearity!
    # (either additive or more general!)
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(Ncomp)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

        # we first apply non-linear function, then causal matrix!
        if NonLin == 'leaky':
            mixedDat = leaky_ReLU(mixedDat, negSlope)
        elif NonLin == 'sigmoid':
            mixedDat = sigmoidAct(mixedDat)
        # apply mixing:
        mixedDat = np.dot(mixedDat, A)

    return {'source': latents, 'obs': mixedDat, 'labels': labels, 'mixing': mixingList, 'var': modMat,
            'BaseCovariance': BaseCovariance}


def gen_TCL_data_ortho(Ncomp, Nlayer, Nsegment, NsegmentObs, source='Laplace', NonLin='leaky', negSlope=.2,
                       varyMean=False, Niter4condThresh=1e4, seed=1):
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
    np.random.seed(seed)
    # check input is correct
    assert NonLin in ['leaky', 'sigmoid']

    # generate non-stationary data:
    Nobs = NsegmentObs * Nsegment  # total number of observations
    labels = np.array([0] * Nobs)  # labels for each observation (populate below)

    # generate data, which we will then modulate in a non-stationary manner:
    if source == 'Laplace':
        dat = np.random.laplace(0, 1, (Nobs, Ncomp))
        dat = scale(dat)  # set to zero mean and unit variance
    elif source == 'Gaussian':
        dat = np.random.normal(0, 1, (Nobs, Ncomp))
        dat = scale(dat)
    else:
        raise Exception("wrong source distribution")

    # get modulation parameters
    modMat = np.random.uniform(0.01, 3, (Ncomp, Nsegment))

    # meanMat = np.random.uniform(0, 5, (Ncomp, Nsegment))
    if varyMean:
        meanMat = np.random.uniform(-3, 3, (Ncomp, Nsegment))
    else:
        meanMat = np.zeros((Ncomp, Nsegment))
    # now we adjust the variance within each segment in a non-stationary manner
    for seg in range(Nsegment):
        segID = range(NsegmentObs * seg, NsegmentObs * (seg + 1))
        dat[segID, :] = np.multiply(dat[segID, :], modMat[:, seg])
        dat[segID, :] = np.add(dat[segID, :], meanMat[:, seg])
        labels[segID] = seg

    # now we are ready to apply the non-linear mixtures:
    mixedDat = np.copy(dat)

    # generate mixing matrices:
    # now we apply layers of non-linearity (just one for now!). Note the order will depend on natural of nonlinearity!
    # (either additive or more general!)
    mixingList = []
    for l in range(Nlayer - 1):
        # generate causal matrix first:
        A = ortho_group.rvs(Ncomp)  # generateUniformMat( Ncomp, condThresh )
        mixingList.append(A)

        # we first apply non-linear function, then causal matrix!
        if NonLin == 'leaky':
            mixedDat = leaky_ReLU(mixedDat, negSlope)
        elif NonLin == 'sigmoid':
            mixedDat = sigmoidAct(mixedDat)
        # apply mixing:
        mixedDat = np.dot(mixedDat, A)

    return {'source': dat, 'obs': mixedDat, 'labels': labels, 'mixing': mixingList, 'var': modMat}


def generate_synthetic_data(data_dim, data_segments, n_obs_seg, n_layer, simulationMethod='TCL', seed=1,
                            one_hot_labels=False, varyMean=False):
    np.random.seed(seed)
    if simulationMethod.lower() == 'tcl':
        dat_all = gen_TCL_data_ortho(Ncomp=data_dim, Nsegment=data_segments, Nlayer=n_layer, NsegmentObs=n_obs_seg,
                                     source='Gaussian', NonLin='leaky', negSlope=.2, seed=seed, varyMean=varyMean)
    elif simulationMethod.lower() == 'imca':
        baseEvals = np.random.rand(data_dim)
        baseEvals /= ((1. / data_dim) * baseEvals.sum())
        baseCov = random_correlation.rvs(baseEvals)

        dat_all = gen_IMCA_data(Ncomp=data_dim, Nsegment=data_segments, Nlayer=n_layer, NsegmentObs=n_obs_seg,
                                NonLin='leaky', negSlope=.2, BaseCovariance=baseCov, seed=seed)
    else:
        raise ValueError('invalid simulation method: {}'.format(simulationMethod))
    x = dat_all['obs']
    if one_hot_labels:
        y = to_one_hot(dat_all['labels'])[0]
    else:
        y = dat_all['labels']
    s = dat_all['source']

    return x, y, s


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # X = gen_nonstationary_data(Ncomp=3, Nsegment=5, Nlayer=3, NsegmentObs=100, NonLin='leaky', negSlope=.2, Niter4condThresh=1e4)

    baseCov = np.array([[1, .6], [.6, 1]])
    X = gen_IMCA_data(Ncomp=2, Nsegment=5, Nlayer=3, BaseCovariance=baseCov, NsegmentObs=100, NonLin='leaky',
                      negSlope=.2, Niter4condThresh=1e4)

    plt.scatter(X['source'][:, 0], X['source'][:, 1])
    plt.title('correlated latents')

    plt.scatter(X['obs'][:, 0], X['obs'][:, 1])
    plt.title('correlated latents')
