import numpy as np


def calc_cond(M):
    """
    Calculates the condition number of a given matrix M using the max-norm
    """
    M_inv = np.linalg.inv(M)
    M_norm = np.linalg.norm(M, ord=np.inf)
    M_inv_norm = np.linalg.norm(M_inv, ord=np.inf)
    cond_num = M_norm * M_inv_norm
    return cond_num


def lu_factorize(A):
    """
    Performs LU-factorization on a square matrix A.
    """

    # Initialize arrays needed
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros(A.shape)
    for k in range(n-1):
        if A[k, k] == 0:
            # Stop if singular
            return

        # Create subdiagonal elements of the k'th column of L
        L[k+1:, k] = A[k+1:, k] / A[k, k]

        for j in range(k+1, n):
            # Apply transformation matrix to the rest of the submatrix
            A[k+1:, j] = A[k+1:, j] - L[k+1:, k] * A[k, j]

        # Assign upper triangular part of A to U
        U[:k+1, k] = A[:k+1, k]

    # Assign last column of A to U
    U[:, -1] = A[:, -1]
    return L, U


def forward_substitute(L, b):
    """
    Performs forward substitution on the lower triangular system Ly=b to solve
    for y. Assumes the diagonal of L is 1.
    """
    y = np.zeros(b.shape)
    for i in range(b.size):
        y[i] = b[i] - L[i, :i].dot(y[:i])
    return y


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
        x[i] = (y[i] - U[i, i:].dot(x[i:]))/U[i, i]
    return x


def linsolve(A, b):
    """
    Solve the linear system Ax=b with LU-decomposition and
    forward/back-substitution.
    """
    # Copy the matrix so as not to change it in place.
    A = A.copy()
    L, U = lu_factorize(A)
    y = forward_substitute(L, b)
    x = back_substitute(U, y)
    return x


def forward_error_bound(E, S, omega, domega=5e-4):
    """
    Calculates the error bound from question b1 with a standard pertubation of
    0.5*10^-3.
    """

    M = E-omega*S
    cond = calc_cond(M)
    num = np.linalg.norm(M, ord=np.inf)
    # den = np.linalg.norm(domega*S, ord=np.inf)
    # Well, S is a diagonal matrix with unity elements (alternating signs,
    # though), so the max-norm is just the multiplier
    den = domega
    return cond * den / num


def solve_alpha(omega, E, S, z):
    """
    Solves the polarization problem using LU-decomposition
    """

    M = E-omega*S
    L, U = lu_factorize(M)
    y = forward_substitute(L, z)
    x = back_substitute(U, y)
    alpha = z.dot(x)
    return alpha


def householder_QR(A):
    """
    Performs householder QR factorization on a rectangular (m,n) matrix, with
    m>n.
    """
    m, n = A.shape
    Q = np.zeros(m, m)
    R = np.zeros(m, n)

    return Q, R


def least_squares(A, b):
    """
    Performs a least-squares fit for the rectangular system Ax=b.
    """
    m, n = A.shape
    x = np.zeros(n)

    return x
