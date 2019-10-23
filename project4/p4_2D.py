import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from f4 import *
from progress.bar import Bar


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


def calc_next_time(p, q, h, dt, params):
    """
    Calculates the next time step for p and q, given the reaction diffusion
    equations. Uses Finite Difference approximations in calculating
    derivatives.

    Assumes ghost nodes have been updated
    """
    Dp, Dq, C, K = params

    # C is kill rate
    # K is feed rate

    # Get needed subarrays for domain nodes:
    N = p.shape[0]-2

    # Inner nodes
    p_i_j = p[1:N+1, 1:N+1]
    q_i_j = q[1:N+1, 1:N+1]

    p_lap = calc_laplace(p, h)
    q_lap = calc_laplace(q, h)

    f = Dp*p_lap + p_i_j*p_i_j*q_i_j + C - (K+1)*p_i_j
    g = Dq*q_lap - p_i_j*p_i_j*q_i_j + K*p_i_j

    p[1:N+1, 1:N+1] = p_i_j + dt*f
    q[1:N+1, 1:N+1] = q_i_j + dt*g

    return p, q


def update_ghosts(p, q):
    """
    Update ghost nodes for p and q arrays.
    We have no-flux Neumann conditions, so dp/dn=dq/dn = 0 on all edges.
    We use a central difference approximation to uphold this
    """

    # Horizontal borders
    p[0] = p[2]
    p[-1] = p[-3]

    # Vertical borders
    p[:, 0] = p[:, 2]
    p[:, -1] = p[:, -3]

    # Horizontal borders
    q[0] = q[2]
    q[-1] = q[-3]

    # Vertical borders
    q[:, 0] = q[:, 2]
    q[:, -1] = q[:, -3]

    return p, q


def simRD(Nx, params, Nt=None, T_end=2000, p0=None, q0=None):
    """
    Simulates reaction-diffusion
    """
    _, _, C, K = params

    # Create the computational grid - a square: [0, 40] X [0, 40]
    x, h = linspace_with_ghosts(0, 40, Nx)
    xx, yy = np.meshgrid(x, x)

    # Calc timestep
    if Nt is None:
        dt = h*h/(4*max(params))
        Nt = np.ceil(T_end/dt).astype(int)
    t = np.linspace(0, T_end, Nt)
    dt = t[1]-t[0]

    # Create p and q arrays:
    p = np.zeros((Nx+2, Nx+2, Nt))
    q = np.zeros((Nx+2, Nx+2, Nt))

    # Populate initial condition - includes ghost nodes
    initial_x = (xx <= 30) * (xx >= 10)
    initial_y = (yy <= 30) * (yy >= 10)
    initial = initial_x * initial_y
    p0 = C + 0.1
    q0 = K/C + 0.2
    p[initial, 0] = p0
    q[initial, 0] = q0

    bar = Bar("Simulating", max=Nt)
    bar.next()
    for k in range(1, Nt):
        # Update domain based on last step:
        p[:, :, k], q[:, :, k] = calc_next_time(p[:, :, k-1], q[:, :, k-1],
                                                h, dt, params)
        # Update ghost nodes
        p[:, :, k], q[:, :, k] = update_ghosts(p[:, :, k], q[:, :, k])
        bar.next()
    bar.finish()

    return p, q, xx, yy, t


def simtest():
    Nx = 100
    params = [1, 8, 4.5, 9]
    # Nt = 120000
    p, q, xx, yy, t = simRD(Nx, params)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(xx, yy, p[:, :, -1])

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(xx, yy, q[:, :, -1])

    plt.show()


def main():
    simtest()


if __name__ == "__main__":
    main()
