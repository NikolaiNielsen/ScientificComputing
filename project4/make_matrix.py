import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from f4 import *


def indexHelper(i, j, N):
    return N * i + j


def create_diffusion_jacobian(p, c, dt, h):
    """
    Create the jacobian for the diffusion system. N points in each direction.
    Ghost nodes around it.
    """

    # Get number of interior points
    N = p.shape[0] - 2
    inner = np.zeros(p.shape)
    inner[1:-1, 1:-1] = 1
    inner = inner == 1
    inner_flat = inner.flatten()
    jac = np.zeros(((N+2)**2, (N+2)**2))
    # Populating inner part of the jacobian
    const1 = 1-2*dt*c/h**2
    const2 = -dt*c/(2*h**2)
    for i in range(1, N+1):
        # sides
        left = indexHelper(np.array([i, i]), np.array([0, 1]), N+2)
        right = indexHelper(np.array([i, i]), np.array([N+1, N]), N+2)
        jac[[left[0], left[0]], left] = [1, -1]
        jac[[right[0], right[0]], right] = [1, -1]

        # inner
        for j in range(1, N+1):
            ij = indexHelper(np.array([i, i+1, i-1, i, i]),
                             np.array([j, j, j, j+1, j-1]), N+2)
            m, ipj, imj, ijp, ijm = ij
            jac[[m, m, m, m, m], ij] = [const1, const2, const2, const2, const2]
    for i in range(0, N+2):
        # Upper and lower
        ij = indexHelper(np.array([0, 1]), np.array([i, i]), N+2)
        ij2 = indexHelper(np.array([N+1, N]), np.array([i, i]), N+2)
        jac[[i, i], ij] = [1, -1]
        jac[[ij2[0], ij2[0]], ij2] = [1, -1]
    return jac


def test_jac():
    Nx = 10
    x, h = linspace_with_ghosts(0, 1, Nx)
    p = np.zeros((Nx+2, Nx+2))
    c = 1
    dt = 1
    jac = create_diffusion_jacobian(p, c, dt, h)
    fig, ax = plt.subplots()
    ax.spy(jac)
    plt.show()


def main():
    test_jac()


if __name__ == "__main__":
    main()
