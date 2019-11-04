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


def RD_jacobian_const(p, params, h, dt):
    """
    Creates the jacobian for the Reaction Diffusion problem.
    """
    Nx = p.shape[0]
    Dp, Dq, C, K = params
    jac = sp.lil_matrix((2*Nx*Nx, 2*Nx*Nx))
    N = jac.shape[0]

    p_const1 = 1+(K+1+4*Dp/h**2)*dt/2
    p_const2 = -dt*Dp/(2*h**2)

    q_const1 = 1+2*dt*Dq/h**2
    q_const2 = -dt*Dq/(2*h**2)
    q_const3 = -dt*K/2

    for i in range(1, Nx-1):
        # top ghosts:
        ptop = indexHelper(0, i, Nx, 0)
        qtop = indexHelper(0, i, Nx, 1)
        jac[ptop, ptop] = 1
        jac[qtop, qtop] = 1

        # Bottom ghosts
        pbot = indexHelper(Nx-1, i, Nx, 0)
        qbot = indexHelper(Nx-1, i, Nx, 1)
        jac[pbot, pbot] = 1
        jac[qbot, qbot] = 1

        for j in range(1, Nx-1):
            # Populate inner nodes:

            # The i's and j's needed for the laplacian
            i_arr = np.array([i, i+1, i-1, i, i])
            j_arr = np.array([j, j, j, j+1, j-1])
            # Converted to vector index
            pij = indexHelper(i_arr, j_arr, Nx, 0)
            qij = indexHelper(i_arr, j_arr, Nx, 1)

            # The current node for p and q
            pm = pij[0]
            qm = qij[0]

            # Laplacian
            jac[[pm]*5, pij] = [p_const1] + [p_const2] * 4
            jac[[qm]*5, qij] = [q_const1] + [q_const2] * 4

            # There is only a constant coupling from p to q.
            jac[qm, pm] = q_const3

    for i in range(0, Nx):
        # left and right nodes.
        pleft = indexHelper(i, 0, Nx, 0)
        qleft = indexHelper(i, 0, Nx, 1)
        pright = indexHelper(i, Nx-1, Nx, 0)
        qright = indexHelper(i, Nx-1, Nx, 1)
        jac[pleft, pleft] = 1
        jac[qleft, qleft] = 1
        jac[pright, pright] = 1
        jac[qright, qright] = 1

    return jac.todia()


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


def f_consts(p, q, params, h, dt):
    Dp, Dq, C, K = params
    alpha = p.copy()
    beta = q.copy()
    pij = p[1:-1, 1:-1]
    qij = q[1:-1, 1:-1]
    plap = calc_laplace(p, h)
    qlap = calc_laplace(q, h)

    alpha[1:-1, 1:-1] = pij + (Dp*plap + pij*pij*qij + C - (K+1)*pij)*dt/2
    beta[1:-1, 1:-1] = qij + (Dq*qlap - pij*pij*qij + K*pij)*dt/2
    f = np.array((alpha, beta)).flatten()
    return f


def objective_function(p, q, params, h, dt, consts):
    """
    """
    N = p.shape[0]
    Dp, Dq, C, K = params
    pcont = p.copy()
    qcont = q.copy()

    pij = p[1:-1, 1:-1]
    qij = q[1:-1, 1:-1]
    plap = calc_laplace(p, h)
    qlap = calc_laplace(q, h)
    pcont[1:-1, 1:-1] = pij - (Dp*plap + pij*pij*qij + C - (K+1)*pij)*dt/2
    qcont[1:-1, 1:-1] = qij - (Dq*qlap - pij*pij*qij + K*pij)*dt/2

    f = np.array((pcont, qcont)).flatten() - consts
    return -f


def CN_next_step(p, q, params, h, dt, maxiter=20, tol=1e-3):
    x = np.array((p, q)).flatten()
    N = p.shape[0]
    consts = f_consts(p, q, params, h, dt)
    jac_const = RD_jacobian_const(p, params, h, dt)

    for k in range(maxiter):
        f = objective_function(p, q, params, h, dt, consts)
        jac_diag = RD_jacobian_diag(p, q, params, h, dt)
        jac = jac_const + jac_diag
        s = splin.spsolve(jac, f)
        x = x + s
        p, q = x.reshape((2, N, N))
        res = np.sum(np.sqrt(s**2))/s.size
        print(res)
        if res <= tol:
            break

    return p, q


def main():
    params = [1, 8, 4.5, 9]
    Dp, Dq, C, K = params
    T_end = 2000
    Nx = 101
    x, h = linspace_with_ghosts(0, 40, Nx)
    xx, yy = np.meshgrid(x, x)

    # Calc timestep
    dt = h*h/(4)
    Nt = np.ceil(T_end/dt).astype(int)
    t = np.linspace(0, T_end, Nt)
    dt = t[1]-t[0]
    print(Nt)

    p_new = np.zeros((Nx+2, Nx+2))
    q_new = np.zeros((Nx+2, Nx+2))

    # Populate initial condition - includes ghost nodes
    initial_x = (xx <= 30) * (xx >= 10)
    initial_y = (yy <= 30) * (yy >= 10)
    initial = initial_x * initial_y
    p0 = C + 0.1
    q0 = K/C + 0.2
    p_new[initial] = p0
    q_new[initial] = q0
    p2, q2 = CN_next_step(p_new, q_new, params, h, dt)


if __name__ == "__main__":
    main()
