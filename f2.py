import numpy as np
import f1


def power_iterate(A, x0=None, max_iter=25, epsilon=1e-6, rayleigh=False):
    """
    Performs power iteration on the matrix A to find the extreme eigenvalue.
    Normalized. With option of using rayleigh quotient.
    """
    n, _ = A.shape
    x = np.random.uniform(size=n)
    for i in range(max_iter):
        y = A@x
        x = y / np.amax(y)

    if not rayleigh:
        y = A@x
        return np.amax(y)
    else:
        res = rayleigh_qt(A, x)
        return res


def gershgorin(A):
    """
    Calculates the Greshgorin discs for a given matrix
    """
    centers = np.diag(A)
    radii = np.sum(np.abs(A), axis=1)-np.abs(centers)
    return centers, radii


def rayleigh_qt(A, x):
    """
    Computes the rayleigh quotient for a matrix and vector
    """
    num = x.dot(A@x)
    den = np.sum(x**2)
    lambda_ = num/den
    return lambda_


def rayleigh_iterate(A, max_iter=10):
    """
    Performs Rayleigh Quotient iteration
    """
    n, _ = A.shape
    x = np.random.uniform(size=n)

    for i in range(max_iter-1):
        sigma = rayleigh_qt(A, x)
        B = A-sigma*np.eye(n)
        y = f1.linsolve(B, x)
        x = y/np.amax(y)
    return rayleigh_qt(A, x)
