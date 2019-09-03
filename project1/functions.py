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
        
        for i in range(k+1, n):
            # Compute column of L
            L[i, k] = A[i, k] / A[k, k]

        for j in range(k+1, n):
            for i in range(k+1, n):
                # Update remaining values of 
                A[i, j] = A[i, j] - L[i, k] * A[k, j]

        # Assign upper triangular part of A to U
        U[:k+1, k] = A[:k+1, k]

    # Assign last column of A to U
    U[:, -1] = A[:, -1]
    return L, U
