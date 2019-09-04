import numpy as np


def cond(M):
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


def forward_substitution(L, b):
    """
    Performs forward substitution on the lower triangular system Ly=b to solve
    for y. Assumes the diagonal of L is 1.
    """
    y = np.zeros(b.shape)
    for i in range(b.size):
        y[i] = b[i] - L[i, :i].dot(y[:i])
    return y
