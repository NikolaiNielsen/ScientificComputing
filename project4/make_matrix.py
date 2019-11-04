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
    else:
        jac = np.zeros(((N)**2, (N)**2))
    # Populating inner part of the jacobian
    const1 = 1+4*dt*c/h**2
    const2 = -dt*c/(h**2)
    for i in range(1, N-1):
        # sides
        left = indexHelper(i, 0, N)
        right = indexHelper(i, N-1, N)
        jac[left, left] = 1
        jac[right, right] = 1

        # inner
        for j in range(1, N-1):
            ij = indexHelper(np.array([i, i+1, i-1, i, i]),
                             np.array([j, j, j, j+1, j-1]), N)
            m, ipj, imj, ijp, ijm = ij
            jac[[m, m, m, m, m], ij] = [const1, const2, const2, const2, const2]
    for i in range(0, N):
        # Upper and lower
        ij = indexHelper(0, i, N)
        ij2 = indexHelper(N-1, i, N)
        jac[ij, ij] = 1
        jac[ij2, ij2] = 1
    return jac


def create_b_vec(p, c, h, dt):
    b = np.zeros(p.shape)
    N = p.shape[0]-2
    lap = calc_laplace(p, h)
    b[:, :] = p
    b[1:-1, 1:-1] = lap*c*0.5*dt + p[1:-1, 1:-1]
    return b.flatten()


def test_2D():
    Nx = 101
    x, h = linspace_with_ghosts(0, 1, Nx)
    xx, yy = np.meshgrid(x, x)
    p = np.zeros((Nx+2, Nx+2))
    c = 1
    dt = h**2/(3*c)
    Nt = 100
    jac = create_diffusion_jacobian(p, c, dt, h, sparse=True)
    # print(jac)
    jac = jac.tocsr()
    sigma = 0.1
    p = np.exp(-((xx-0.5)**2/(2*sigma**2) + (yy-0.5)**2/(2*sigma**2)))
    p[0] = 0
    p[-1] = 0
    p[:, 0] = 0
    p[:, -1] = 0
    # p0 = p.copy()
    # p[Nx//2, Nx//2] = 10
    pold = p.copy()

    xx, yy = np.meshgrid(x, x)
    for i in range(1, Nt):
        # b = create_b_vec(p, c, h, dt)
        implicit = splin.spsolve(jac, p.flatten()).reshape((Nx+2, Nx+2))
        # explicit = create_b_vec(p, c, h, dt).reshape((Nx+2, Nx+2))
        p = (implicit) # + explicit)/2
        p[0] = 0
        p[-1] = 0
        p[:, 0] = 0
        p[:, -1] = 0

    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # ax.plot_surface(xx, yy, p)
    fig, ax = plt.subplots()
    ax.plot(p[50])
    print(p[50, 50])
    ax.plot(pold[50])
    plt.show()


def test_1D():
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
        p = np.linalg.solve(A, p)
        P.append(p.copy())

    p = np.array(P)
    t = np.arange(Nt) * dt

    xx, tt = np.meshgrid(x, t)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(tt, xx, p)

    plt.show()


def test_mat():
    Nx = 3
    x, h = linspace_with_ghosts(0, 1, Nx)
    xx, yy = np.meshgrid(x, x)
    p = np.zeros((Nx+2, Nx+2))
    c = 1
    dt = 1
    jac = create_diffusion_jacobian(p, c, dt, h)
    print(jac.toarray())
    p[2, 2] = 10
    sol = splin.spsolve(jac, p.flatten()).reshape((5, 5))
    print(sol)
    # fig, ax = plt.subplots()


def main():
    test_2D()


if __name__ == "__main__":
    main()
