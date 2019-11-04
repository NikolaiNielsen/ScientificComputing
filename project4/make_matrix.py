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


def create_implicit_diffusion_jacobian(p, c, dt, h, sparse=True):
    """
    Create the jacobian for the diffusion system. N points in each direction.
    With ghost nodes.
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


def CN_diffusion_jacobian(p, c, dt, h, sparse=True):
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
    const1 = 1+2*dt*c/h**2
    const2 = -dt*c/(2*h**2)
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


def CN_coupled_diffusion_jac(p, params, h, dt):
    """
    Creates the jacobian for the coupled diffusion problem.
    """
    n = p.shape[0]
    Dp, Dq, cp, cq = params
    jac = sp.lil_matrix((2*n*n, 2*n*n))
    N = jac.shape[0]

    p_const1 = 1+2*dt*Dp/h**2
    p_const2 = -dt*Dp/(2*h**2)
    p_const3 = -cp*dt/2

    q_const1 = 1+2*dt*Dq/h**2
    q_const2 = -dt*Dq/(2*h**2)
    q_const3 = -cq*dt/2

    for i in range(1, n-1):
        # top ghosts:
        ptop = indexHelper(0, i, n, 0)
        qtop = indexHelper(0, i, n, 1)
        jac[ptop, ptop] = 1
        jac[qtop, qtop] = 1

        # Bottom ghosts
        pbot = indexHelper(n-1, i, n, 0)
        qbot = indexHelper(n-1, i, n, 1)
        jac[pbot, pbot] = 1
        jac[qbot, qbot] = 1

        for j in range(1, n-1):
            # Populate inner nodes:

            # The i's and j's needed for the laplacian
            i_arr = np.array([i, i+1, i-1, i, i])
            j_arr = np.array([j, j, j, j+1, j-1])
            # Converted to vector index
            pij = indexHelper(i_arr, j_arr, n, 0)
            qij = indexHelper(i_arr, j_arr, n, 1)

            # The current node for p and q
            pm = pij[0]
            qm = qij[0]

            # Laplacian
            jac[[pm]*5, pij] = [p_const1] + [p_const2] * 4
            jac[[qm]*5, qij] = [q_const1] + [q_const2] * 4

            # Coupling between p and q:
            jac[pm, qm] = p_const3
            jac[qm, pm] = q_const3

    for i in range(0, n):
        # left and right nodes.
        pleft = indexHelper(i, 0, n, 0)
        qleft = indexHelper(i, 0, n, 1)
        pright = indexHelper(i, n-1, n, 0)
        qright = indexHelper(i, n-1, n, 1)
        jac[pleft, pleft] = 1
        jac[qleft, qleft] = 1
        jac[pright, pright] = 1
        jac[qright, qright] = 1

    return jac


def coupled_diffusion_b_vec(p, q, params, h, dt):
    """
    Calculates the b-vector for the coupled diffusion problem.
    """
    Dp, Dq, cp, cq = params

    pvec = p.copy()
    plap = calc_laplace(p, h)
    pvec[1:-1, 1:-1] = p[1:-1, 1:-1] + plap * Dp * dt / 2 + cp * q[1:-1, 1:-1]

    qvec = q.copy()
    qlap = calc_laplace(q, h)
    qvec[1:-1, 1:-1] = q[1:-1, 1:-1] + qlap * Dq * dt / 2 + cq * p[1:-1, 1:-1]

    b = np.array((pvec, qvec)).flatten()
    return b


def diffusion_b_vec(p, c, h, dt):
    b = np.zeros(p.shape)
    N = p.shape[0]-2
    lap = calc_laplace(p, h)
    b[:, :] = p
    b[1:-1, 1:-1] = lap*c*0.5*dt + p[1:-1, 1:-1]
    return b.flatten()


def test_implicit_diffusion():
    Nx = 101
    x, h = linspace_with_ghosts(0, 1, Nx)
    xx, yy = np.meshgrid(x, x)
    p = np.zeros((Nx+2, Nx+2))
    c = 1
    dt = h**2/(3*c)
    Nt = 100
    jac = create_implicit_diffusion_jacobian(p, c, dt, h, sparse=True)
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
        p = implicit
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


def test_CN_diffusion():
    # Initialize grid
    Nx = 101
    Nxp = Nx + 2
    x, h = linspace_with_ghosts(0, 1, Nx)
    xx, yy = np.meshgrid(x, x)

    # Initialize initial condition
    sigma = 0.1
    # p = np.exp(-((xx-0.5)**2/(2*sigma**2) + (yy-0.5)**2/(2*sigma**2)))
    p = np.zeros(xx.shape)
    p[Nxp//2, Nxp//2] = 10
    p[0] = 0
    p[-1] = 0
    p[:, 0] = 0
    p[:, -1] = 0

    pold = p.copy()
    # Set parameters
    c = 1
    dt = h**2/(2*c)
    Nt = 100
    jac = create_implicit_diffusion_jacobian(p, c, dt, h, sparse=True)
    jac = jac.tocsr()

    p_mid = [p[Nxp//2]]

    for i in range(1, Nt):
        b = diffusion_b_vec(p, c, h, dt)
        p = splin.spsolve(jac, b).reshape(xx.shape)
        p[0] = 0
        p[-1] = 0
        p[:, 0] = 0
        p[:, -1] = 0
        p_mid.append(p[Nxp//2])

    t = dt*np.arange(Nt)
    X, T = np.meshgrid(x, t)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(X, T, np.array(p_mid))
    plt.show()


def test_1D_CN_diffusion():
    # Per https://pycav.readthedocs.io/en/latest/api/pde/crank_nicolson.html
    Nx = 101
    # x, h = linspace_with_ghosts(0, 1, Nx)
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


def test_matrix():
    Nx = 6
    x, h = linspace_with_ghosts(0, 5, Nx)
    p = np.outer(x, x)
    J = CN_coupled_diffusion_jac(p, [1, 1, 1, 1], 1, 1)
    fig, ax = plt.subplots()
    ax.spy(J)
    plt.show()


def main():
    test_matrix()


if __name__ == "__main__":
    main()
