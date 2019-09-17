import numpy as np


def power_iteration(A, max_iter=20, normalized=True):
    """
    Performs power iteration on the matrix A to find the extreme eigenvalue.
    Normalized by default.
    """
    n, _ = A.shape
    x = np.random.uniform(size=n)
    for i in range(max_iter):
        y = A@x
        x = y / np.amax(y)
    else:
        y = A@x
        return np.amax(y)


def gershgorin_disks(A):
    """
    Calculates the Greshgorin discs for a given matrix
    """
    centers = np.diag(A)
    row_sums = np.sum(np.abs(A), axis=1)-np.abs(centers)
    return centers, row_sums
