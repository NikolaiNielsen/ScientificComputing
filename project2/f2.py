import numpy as np


def householder_QR(A, inline=True):
    """
    Performs householder QR factorization on a rectangular (m,n) matrix, with
    m>n.
    """
    A = A.copy() if not inline else A
    m, n = A.shape
    Q = np.identity(m)
    # R = np.zeros(m, n)
    H_list = []
    for i in range(n):
        a = A[:, i]
        alpha = -np.sign(a[i]) * np.sqrt(np.sum(a[i:]**2))
        v = np.zeros(m)
        v[i:] = a[i:]
        v[i] = v[i] - alpha
        beta = v.dot(v)
        if beta == 0:
            continue
        else:
            H = np.identity(m) - 2 * np.outer(v, v) / beta
            Q = H@Q
            gammaVec = v.dot(A[:, i:])
            A[:, i:] = A[:, i:] - (2*np.outer(v, gammaVec)/beta)
    R = A
    return Q.T, R


def back_substitute(U, y):
    """
    Performs backwards substitution on the upper triangular system Ux=y to
    solve for x. Does not assume a diagonal of U of 1.
    """

    # We could actually combine the forward and backwards substitution, if we
    # define i -> -i for backward substitution, and run the loop in the
    # "forward" direction.
    x = np.zeros(y.shape)
    for i in reversed(range(y.size)):
        if U[i, i] == 0:
            # Return None if singular
            return None
        x[i] = (y[i] - U[i, i:].dot(x[i:]))/U[i, i]
    return x


def qr_solve(A, b, inline=True):
    """
    Performs a least-squares fit for the rectangular system Ax=b, using
    Householder QR-factorization and backsubstitution on the system Rx=c1
    """
    m, n = A.shape
    Q, R = householder_QR(A, inline=inline)
    b2 = Q.T@b
    x = back_substitute(R[:n], b2[:n])

    return x


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


def power_iterate(A, x0=None, max_iter=25, epsilon=1e-6, shift=0.):
    """
    Performs power iteration on the matrix A to find the extreme eigenvector.
    Normalized. With option of using rayleigh quotient.
    """
    n, _ = A.shape

    # Set shift, if needed
    B = A - np.eye(n)*shift

    # Choose random x0 if not given one
    x = np.random.uniform(size=n) if x0 is None else x0

    lambda_last = rayleigh_qt(A, x)
    for i in range(max_iter):
        # Calculate next iteration
        y = B@x
        x = y / np.amax(y)

        # Calculate approximate eigenvalue, check for convergence
        lambda_new = rayleigh_qt(A, x)
        res = abs((lambda_new-lambda_last)/(lambda_last))
        if res <= epsilon:
            break
        lambda_last = lambda_new

    return x, i+1


def rayleigh_iterate(A, x0=None, shift=None, epsilon=1e-6, max_iter=10):
    """
    Performs Rayleigh Quotient iteration
    """
    n, _ = A.shape
    x = np.random.uniform(size=n) if x0 is None else x0

    if shift is not None:
        x, _ = inverse_interate(A, x0=x, shift=shift)
        if x is None:
            # As with inverse iteration - x is None if QR-factorization yielded
            # a singular matrix.
            return None, None

    # Calc approx eigenvalue and shift matrix
    sigma = rayleigh_qt(A, x)
    for i in range(max_iter):
        B = A-sigma*np.eye(n)

        # Find new eigenvector
        y = qr_solve(B, x)
        x = y/np.amax(y)

        # Test for convergence
        sigma_new = rayleigh_qt(A, x)
        res = abs((sigma_new-sigma)/(sigma))
        if res <= epsilon:
            break
        sigma = sigma_new
    return x, i+1


def inverse_interate(A, x0=None, shift=0., epsilon=1e-6, max_iter=5):
    n, _ = A.shape
    B = A - np.eye(n)*shift
    x = np.random.uniform(size=n) if x0 is None else x0
    lambda_last = rayleigh_qt(A, x)

    for i in range(max_iter):
        # Find new eigenvector
        y = qr_solve(B, x, inline=False)

        if y is None:
            # y is none, if QR yielded a singular matrix. Return None
            return None, None

        x = y/np.amax(y)

        # Test for convergence
        lambda_new = rayleigh_qt(A, x)
        res = abs((lambda_new-lambda_last)/(lambda_last))
        if res <= epsilon:
            break
        lambda_new = lambda_last
    return x, i+1


def find_unique(a, b=None, rtol=1e-5, atol=1e-8):
    """
    Finds values of a that are approximately unique
    """
    unique = []
    if isinstance(a, list):
        a = np.array(a)

    while len(a) > 1:
        close_to = np.isclose(a[0], a, rtol, atol)
        avg = np.mean(a[close_to])
        a = a[~close_to]
        if b is not None:
            unique.append((avg, b[0]))
            b = b[~close_to]
        else:
            unique.append(avg)

    return sorted(unique, reverse=True)
