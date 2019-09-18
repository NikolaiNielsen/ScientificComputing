import numpy as np
import f1


def power_iterate(A, x0=None, max_iter=25, epsilon=1e-6, rayleigh=False):
    """
    Performs power iteration on the matrix A to find the extreme eigenvector.
    Normalized. With option of using rayleigh quotient.
    """
    n, _ = A.shape

    # Choose random x0 if not given one
    x = np.random.uniform(size=n) if x0 is None else x0

    lambda_last = rayleigh_qt(A, x)
    for i in range(max_iter):
        # Calculate next iteration
        y = A@x
        x = y / np.amax(y)

        # Calculate approximate eigenvalue, check for convergence
        lambda_new = rayleigh_qt(A, x)
        res = abs((lambda_new-lambda_last)/(lambda_last))
        if res <= epsilon:
            break
        lambda_last = lambda_new

    return x, i+1


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
