import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from f3 import *


EPSILON = 0.997
SIGMA = 3.401
A = 4*EPSILON*SIGMA**12
B = 4*EPSILON*SIGMA**6


def potential(r, r0=0):
    d = r.shape[0]
    r0 = r0.reshape((d, 1))
    R = r - r0
    R = np.sum(R**2, axis=0)
    V = A/R**6 - B/R**3
    return V


def potentials(r, r0):
    """
    Assumes r is a (3,) array and r0 is a (3,n)-array.
    Outputs a scalar for the potential at point r
    """
    n = r0.shape[0]
    r = r.reshape((n, -1))
    R = r-r0
    R = np.sum(R**2, axis=0)
    V = A/R**6 - B/R**3
    V = np.sum(V)
    return V


def pot_grad(r, r0):
    n = r0.shape[0]
    r = r.reshape((n, -1))
    R = r-r0
    R2 = np.sum(R**2, axis=0)
    factor = 12*A/(R2**7) - 6*B/(R2**4)
    grad_per_atom = factor * R
    total_grad = grad_per_atom.sum(axis=1)
    return total_grad


def q1():
    r1 = np.linspace(1.65, 10.0, num=500)
    r0 = 0
    p1 = potential(r1, r0)
    fig, (ax, ax2) = plt.subplots(nrows=2)
    ax.plot(r1, p1)
    ax.plot(r1[p1 < 0], p1[p1 < 0])
    r2 = np.linspace(2, 4.5, 100)
    p2 = potential(r2, r0)
    ax2.plot(r2, p2)
    # ax2.plot(r2[p2 < 0], p2[p2 < 0])
    ax2.set_ylim(-1.5, 7.5)

    x = newton_raphson(potential, x0=2)
    fx = potential(x)

    x2 = bisection(potential, 2, 4)
    x3 = secant(potential, 1.65, 2)[-1]
    x4 = inverse_quadratic(potential, 1, 2, 3)[-1]
    print(x[-1])
    print(x2)
    print(x3)
    print(x4)
    ax2.plot(x, fx, '-o')
    plt.show()


def test_pot():
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    r0 = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]]).T
    d, spacing, N = 1, 0.1, 100
    lin = np.linspace(spacing, d-spacing, N)
    x, y = np.meshgrid(lin, lin)
    r = np.array((x.flatten(), y.flatten()))
    p = np.zeros(x.size)
    for n, point in enumerate(r.T):
        p[n] = potentials(point, r0)
    # Fx = potentials(r, r0).reshape((N, N))
    p = p.reshape((N, N))
    ax.plot_surface(x, y, p)
    plt.show()


def q3():
    np.random.seed(42)
    data = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    r0 = data.T
    # r_max = np.amax(r0, axis=1)
    # r_min = np.amin(r0, axis=1)
    # r_start = np.random.uniform(r_min, r_max)
    r_start = np.array([0, 0.02, 0.02])

    pot_grad(r_start, r0)
    # def potential_proper(r):
    #     return potentials(r, r0)

    # r = conjugate_gradient(potential_proper, r_start, alpha_0=0.01)
    # ax.scatter(r0[0], r0[1], r0[2])
    # print(len(r))
    # plt.show()


def main():
    q3()


if __name__ == "__main__":
    main()
