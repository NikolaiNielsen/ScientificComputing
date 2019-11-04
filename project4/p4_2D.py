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


def simRD(Nx, params, Nt=None, T_end=2000, return_all=False, tolf=1e-9):
    """
    Simulates reaction-diffusion
    """
    _, _, C, K = params

    # Create the computational grid - a square: [0, 40] X [0, 40]
    x, h = linspace_with_ghosts(0, 40, Nx)
    xx, yy = np.meshgrid(x, x)

    # Calc timestep
    if Nt is None:
        dt = h*h/(4*max(params)*2*1.2)
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

    p_all = [p_new]
    q_all = [q_new]

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
        if return_all:
            p_all.append(p_new.copy())
            q_all.append(q_new.copy())

        # if tolf > residuals[0, k]:
        #     break
        bar.next()
    bar.finish()

    if return_all:
        return p_all, q_all, xx, yy, residuals
    # Only return computational domain:
    p = p_new[1:Nx+1, 1:Nx+1]
    q = q_new[1:Nx+1, 1:Nx+1]
    xx = xx[1:Nx+1, 1:Nx+1]
    yy = yy[1:Nx+1, 1:Nx+1]
    return p, q, xx, yy, residuals


def simtest():
    Nx = 41
    params = [1, 8, 4.5, 9]
    K = [7, 8, 9, 10, 11, 12]
    file1 = [f'RD_K{i}_p' for i in K]
    file2 = [f'RD_K{i}_q' for i in K]
    file3 = [f'RD_K{i}_res' for i in K]
    for k, f1, f2, f3 in zip(K, file1, file2, file3):
        params[-1] = k
        p, q, xx, yy, res = simRD(Nx, params, T_end=2000)
        np.save(f1, p)
        np.save(f2, q)
        np.save(f3, res)


def check_results():
    K = list(range(7, 13))
    names = [f'RD_K{k}' for k in K]
    Nx = 41
    x = np.linspace(0, 40, Nx)
    xx, yy = np.meshgrid(x, x)
    cmap = 'coolwarm'
    for n, name in enumerate(names):
        q = np.load(name + '_q.npy')
        p = np.load(name + '_p.npy')
        # max_ = max(np.amax(p), np.amax(q))
        # min_ = min(np.amin(p), np.amin(q))

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(7.5, 4))
        im1 = ax1.imshow(p, cmap=cmap, origin='lower')
        im2 = ax2.imshow(q, cmap=cmap, origin='lower')
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax1.set_title(f'$p(x,y,t=2000)$, $K=${K[n]}')
        ax2.set_title(f'$q(x,y,t=2000)$, $K=${K[n]}')
        bounds1 = ax1.get_position().bounds
        bounds2 = ax2.get_position().bounds
        fig.subplots_adjust(bottom=0.2)
        cbar_ax1 = fig.add_axes([bounds1[0], 0.07, bounds1[2], 0.05])
        cbar_ax2 = fig.add_axes([bounds2[0], 0.07, bounds2[2], 0.05])
        fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
        fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
        fig.savefig(name + '.pdf')


def check_residuals():
    K = [7, 8, 9, 10, 11, 12]
    names = [f'RD_K{k}' for k in K]
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 6))
    ax = ax.flatten()
    for n, name in enumerate(names):
        res = np.load(name + "_res.npy")
        ax[n].plot(res[0])
        ax[n].set_yscale('log')
        ax[n].set_title(name)
    fig.tight_layout()
    plt.show()


def test_small():
    Nx = 41
    params = [1, 8, 4.5, 7]
    p, q, xx, yy, res = simRD(Nx, params, T_end=2000)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    ax1.imshow(p)
    ax2.plot(res[0])
    ax2.set_yscale('log')
    plt.show()


def main():
    # simtest()
    check_results()
    check_residuals()


if __name__ == "__main__":
    main()
