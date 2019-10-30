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

    return p, q, f, g


def update_ghosts(p, q):
    """
    Update ghost nodes for p and q arrays.
    We have no-flux Neumann conditions, so dp/dn=dq/dn = 0 on all edges.
    We use a central difference approximation to uphold this
    """

    # Horizontal borders
    p[0] = p[1]
    p[-1] = p[-2]

    # Vertical borders
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]

    # Horizontal borders
    q[0] = q[1]
    q[-1] = q[-2]

    # Vertical borders
    q[:, 0] = q[:, 1]
    q[:, -1] = q[:, -2]

    return p, q


def simRD(Nx, params, Nt=None, T_end=2000):
    """
    Simulates reaction-diffusion
    """
    _, _, C, K = params

    # Create the computational grid - a square: [0, 40] X [0, 40]
    x, h = linspace_with_ghosts(0, 40, Nx)
    xx, yy = np.meshgrid(x, x)

    # Calc timestep
    if Nt is None:
        dt = h*h/(4*max(params)*1.2)
        Nt = np.ceil(T_end/dt).astype(int)
    t = np.linspace(0, T_end, Nt)
    dt = t[1]-t[0]

    # Create p and q arrays:
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

    p_old = p_new.copy()
    q_old = q_new.copy()

    residuals = np.zeros((2, Nt))

    bar = Bar("Simulating", max=Nt)
    bar.next()
    for k in range(1, Nt):
        # Update domain based on last step:
        p_new, q_new, f, g = calc_next_time(p_old, q_old, h, dt, params)
        residuals[0, k] = np.sum(f**2)/(Nx**2)
        residuals[1, k] = np.sum(g**2)/(Nx**2)
        # Update ghost nodes
        p_new, q_new = update_ghosts(p_new, q_new)
        # propagate solution
        p_old, q_old = p_new, q_new
        bar.next()
    bar.finish()

    # Only return computational domain:
    p = p_new[1:Nx+1, 1:Nx+1]
    q = q_new[1:Nx+1, 1:Nx+1]
    xx = xx[1:Nx+1, 1:Nx+1]
    yy = yy[1:Nx+1, 1:Nx+1]
    return p, q, xx, yy, residuals


def simtest():
    Nx = 201
    params = [1, 8, 4.5, 9]
    K = [11, 12]
    file1 = [f'BigRD_K{i}_p' for i in K]
    file2 = [f'BigRD_K{i}_q' for i in K]
    file3 = [f'BigRD_K{i}_res' for i in K]
    for k, f1, f2, f3 in zip(K, file1, file2, file3):
        params[-1] = k
        p, q, xx, yy, res = simRD(Nx, params, T_end=2000)
        np.save(f1, p)
        np.save(f2, q)
        np.save(f3, res)


def check_results():
    K = list(range(7, 13))
    names = [f'BigRD_K{k}' for k in K]
    Nx = 201
    x = np.linspace(0, 40, Nx)
    xx, yy = np.meshgrid(x, x)
    cmap = 'coolwarm'
    for name in names:
        q = np.load(name + '_q.npy')
        p = np.load(name + '_p.npy')
        # max_ = max(np.amax(p), np.amax(q))
        # min_ = min(np.amin(p), np.amin(q))

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(7.5, 4))
        im1 = ax1.contour(xx, yy, p, cmap=cmap)
        im2 = ax2.contour(xx, yy, q, cmap=cmap)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax1.set_title('$p(x,y,t=2000)$')
        ax2.set_title('$q(x,y,t=2000)$')
        bounds1 = ax1.get_position().bounds
        bounds2 = ax2.get_position().bounds
        fig.subplots_adjust(bottom=0.2)
        cbar_ax1 = fig.add_axes([bounds1[0], 0.07, bounds1[2], 0.05])
        cbar_ax2 = fig.add_axes([bounds2[0], 0.07, bounds2[2], 0.05])
        fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
        fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
        fig.savefig(name + '.pdf')


def main():
    # simtest()
    check_results()


if __name__ == "__main__":
    main()
