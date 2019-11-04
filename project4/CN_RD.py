import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from f4 import *


def calc_laplace(z, dx):

    N = z.shape[0] - 2
    # Inner nodes
    z_i_j = z[1:N+1, 1:N+1]

    # Shifted nodes, for the laplacian
    z_im_j = z[:N, 1:N+1]
    z_ip_j = z[2:, 1:N+1]
    z_i_jm = z[1:N+1, :N]
    z_i_jp = z[1:N+1, 2:]

    z_lap = (z_i_jm + z_i_jp + z_im_j + z_ip_j - 4*z_i_j)/(dx*dx)

    return z_lap


def indexHelper(i, j, N, n=0):
    """
    Convert matrix-indexing (i,j) to vector-indexing (m). Assumes a
    (n+1)N x n(+1)N matrix.
    """
    return N**2 * n + N * i + j


def RD_jacobian_const(Nx, params, h, dt):
    """
    """
    pass


def RD_jacobian_diag(p, q, params, h, dt):
    """
    """
    N = p.shape[0]
    Dp, Dq, C, K = params
    p_contrib = np.zeros(p.shape)
    p_contrib[1:-1, 1:-1] = -dt * p[1:-1, 1:-1] * q[1:-1, 1:-1]
    q_contrib = np.zeros(q.shape)
    q_contrib[1:-1, 1:-1] = dt/2 * p[1:-1, 1:-1] * p[1:-1, 1:-1]
    diag1 = np.array((p_contrib, q_contrib)).flatten()
    diag2 = p_contrib.flatten()
    diag3 = q_contrib.flatten()
    offsets = [0, -N*N, N*N]
    jac = sp.diags([diag1, diag2, diag3], offsets)
    return jac


def main():
    p = np.ones((4, 4))
    q = np.ones((4, 4))

    params = [1, 1, 1, 1]
    dt = 1
    h = 1
    np.set_printoptions(threshold=np.inf)
    jac = RD_jacobian_diag(p, q, params, h, dt)
    print(jac.toarray())
    fig, ax = plt.subplots()
    ax.spy(jac)
    plt.show()


if __name__ == "__main__":
    main()
