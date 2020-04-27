"""Preprocessing"""

import numpy as np


# ============================================================
# ============================================================
def pca(x, num_comp=None, params=None, zerotolerance=1e-7):
    """Apply PCA whitening to data.
    Args:
        x: data. 2D ndarray [num_comp, num_data]
        num_comp: number of components
        params: (option) dictionary of PCA parameters {'mean':?, 'W':?, 'A':?}. If given, apply this to the data
        zerotolerance: (option)
    Returns:
        x: whitened data
        parms: parameters of PCA
            mean: subtracted mean
            W: whitening matrix
            A: mixing matrix
    """
    # print("PCA...")

    # Dimension
    if num_comp is None:
        num_comp = x.shape[0]
    # print("    num_comp={0:d}".format(num_comp))

    # From learned parameters --------------------------------
    if params is not None:
        # Use previously-trained model
        print("    use learned value")
        data_pca = x - params['mean']
        x = np.dot(params['W'], data_pca)

    # Learn from data ----------------------------------------
    else:
        # Zero mean
        xmean = np.mean(x, 1).reshape([-1, 1])
        x = x - xmean

        # Eigenvalue decomposition
        xcov = np.cov(x)
        d, V = np.linalg.eigh(xcov)  # Ascending order
        # Convert to descending order
        d = d[::-1]
        V = V[:, ::-1]

        zeroeigval = np.sum((d[:num_comp] / d[0]) < zerotolerance)
        if zeroeigval > 0:  # Do not allow zero eigenval
            raise ValueError

        # Calculate contribution ratio
        contratio = np.sum(d[:num_comp]) / np.sum(d)
        # print("    contribution ratio={0:f}".format(contratio))

        # Construct whitening and dewhitening matrices
        dsqrt = np.sqrt(d[:num_comp])
        dsqrtinv = 1 / dsqrt
        V = V[:, :num_comp]
        # Whitening
        W = np.dot(np.diag(dsqrtinv), V.transpose())  # whitening matrix
        A = np.dot(V, np.diag(dsqrt))  # de-whitening matrix
        x = np.dot(W, x)

        params = {'mean': xmean, 'W': W, 'A': A}

        # Check
        datacov = np.cov(x)

    return x, params
