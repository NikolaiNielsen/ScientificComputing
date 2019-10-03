import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist
from f3 import *
from progress.bar import Bar


EPSILON = 0.997
SIGMA = 3.401
A = 4*EPSILON*SIGMA**12
B = 4*EPSILON*SIGMA**6


def potential(R2, A=A, B=B):
    """
    Calculates the total interatomic potential (assuming Lennard Jones
    potential). Takes all interatomic distances squared as input.
    """
    V = np.sum(A/R2**6 - B/R2**3, axis=0)
    return V


def potential_total(r, A=A, B=B):
    """
    Calculates the total interatomic potential (assuming Lennard Jones
    potential). Takes all absolute coordinates as input
    """
    R2 = pdist(r, metric='sqeuclidean')
    V = np.sum(A/R2**6 - B/R2**3)
    return V


def get_gradient(f, r, h=1e-4, normalize=True):
    """
    Calculates the gradient of V_total
    """
    N = r.size//3
    grad = np.zeros(r.shape)
    for i in range(N):
        r_copy = r.copy()
        x, y, z = r[i]
        variations = np.array([[x+h, y, z],
                               [x-h, y, z],
                               [x, y+h, z],
                               [x, y-h, z],
                               [x, y, z+h],
                               [x, y, z-h]])
        varied_potentials = []
        for j in variations:
            r_copy[i] = j
            varied_potentials.append(f(r_copy))
        dx = (varied_potentials[0] - varied_potentials[1])/(2*h)
        dy = (varied_potentials[2] - varied_potentials[3])/(2*h)
        dz = (varied_potentials[4] - varied_potentials[5])/(2*h)
        grad[i] = [dx, dy, dz]
    if normalize:
        grad = grad/abs(np.max(grad))
    return grad


def test_gradient():
    def pot(r):
        return potential_total(r, 1, 1)

    r = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0.]])

    AnalyticGrad = np.array([[6, 6, 0],
                             [-183/32, -9/32, 0],
                             [-9/32, -183/32, 0]])

    numericalGrad = get_gradient(pot, r, normalize=False)
    res = AnalyticGrad - numericalGrad
    print(numericalGrad)
    print(res)


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


def q3():
    data = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    x = conjugate_gradient(potential_total, data, g=get_gradient,
                           max_iter=10000, epsilon=1e-3)
    # N = len(x)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # ax = ax.flatten()
    # # print(x[0]-data)
    # for i in range(N):
    #     ax[i].scatter(x[i].T[0], x[i].T[1], x[i].T[2])

    # plt.show()
    # writer = anim.FFMpegWriter(fps=60)
    # bar = Bar('Writing movie', max=len(x))

    # dpi = 200
    # outfile = 'movie.mp4'
    # with writer.saving(fig, outfile, dpi):
    #     for X in x:
    #         X = X.T
    #         ax.scatter(X[0], X[1], X[2])
    #         writer.grab_frame()
    #         ax.clear()
    #         bar.next()
    # bar.finish()
    x = x[-1].T
    ax.scatter(x[0], x[1], x[2])
    plt.show()


def main():
    q3()


if __name__ == "__main__":
    main()
