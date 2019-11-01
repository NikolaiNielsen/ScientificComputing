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


def indexHelper(i, j, N):
    return N * i + j


def create_diffusion_jacobian(p, c, dt, h, sparse=True):
    """
    Create the jacobian for the diffusion system. N points in each direction.
    Ghost nodes around it.
    """

    # Get number of points
    N = p.shape[0]
    if sparse:
        jac = sp.lil_matrix((N**2, N**2))
    # jac = np.zeros(((N)**2, (N)**2))
    # Populating inner part of the jacobian
    const1 = 1-2*dt*c/h**2
    const2 = -dt*c/(2*h**2)
    for i in range(1, N-1):
        # sides
        left = indexHelper(np.array([i, i]), np.array([0, 1]), N)
        right = indexHelper(np.array([i, i]), np.array([N-1, N-2]), N)
        jac[[left[0], left[0]], left] = [1, -1]
        jac[[right[0], right[0]], right] = [1, -1]

        # inner
        for j in range(1, N-1):
            ij = indexHelper(np.array([i, i+1, i-1, i, i]),
                             np.array([j, j, j, j+1, j-1]), N)
            m, ipj, imj, ijp, ijm = ij
            jac[[m, m, m, m, m], ij] = [const1, const2, const2, const2, const2]
    for i in range(0, N):
        # Upper and lower
        ij = indexHelper(np.array([0, 1]), np.array([i, i]), N)
        ij2 = indexHelper(np.array([N-1, N-2]), np.array([i, i]), N)
        jac[[i, i], ij] = [1, -1]
        jac[[ij2[0], ij2[0]], ij2] = [1, -1]
    return jac


def create_b_vec(p, c, h, dt):
    b = np.zeros(p.shape)
    N = p.shape[0]-2
    lap = calc_laplace(p, h)
    b[:, :] = p
    b[1:-1, 1:-1] = lap*c*0.5*dt + p[1:-1, 1:-1]
    return b.flatten()


def test_jac():
    Nx = 101
    x, h = linspace_with_ghosts(0, 1, Nx)
    p = np.zeros((Nx+2, Nx+2))
    c = 1
    dt = 0.001
    Nt = 10
    jac = create_diffusion_jacobian(p, c, dt, h)
    jac = jac.tocsr()
    p[x == 0.5, x == 0.5] = 1/h**2
    pold = p.copy()

    xx, yy = np.meshgrid(x, x)
    b = create_b_vec(pold, c, h, dt)
    p = splin.spsolve(jac, b).reshape((Nx+2, Nx+2))

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(xx, yy, p)
    plt.show()


def test_diffusion():
    # Per https://pycav.readthedocs.io/en/latest/api/pde/crank_nicolson.html
    Nx = 101
    x, h = linspace_with_ghosts(0, 1, Nx)
    x = np.linspace(0, 1, Nx)
    h = x[1] - x[0]
    c = 1
    dt = 0.005
    Nt = 1000
    s = c*dt/h**2
    A1 = np.diag((1+s)*np.ones(Nx))
    A2 = np.diag((-s/2)*np.ones(Nx-1), 1)
    A3 = np.diag((-s/2)*np.ones(Nx-1), -1)

    B1 = np.diag((1-s)*np.ones(Nx))
    B2 = np.diag((s/2)*np.ones(Nx-1), 1)
    B3 = np.diag((s/2)*np.ones(Nx-1), -1)

    A = A1 + A2 + A3
    B = B1 + B2 + B3
    # B[0, 0] = 0
    # B[-1, -1] = 0
    # A[0, 0] = 1
    # A[0, 1] = -1
    # A[-1, -1] = 1
    # A[-1, -2] = -1
    p = 1/np.sqrt(2*np.pi*(0.25**2)) * np.exp(-(x-0.5)**2/(2*0.25**2))
    P = [p.copy()]
    for i in range(1, Nt):
        p = np.linalg.solve(A, B@p)
        P.append(p.copy())

    p = np.array(P)
    t = np.arange(Nt) * dt

    xx, tt = np.meshgrid(x, t)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(tt, xx, p)

    plt.show()


def main():
    test_jac()


if __name__ == "__main__":
    main()
