import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import fmin_cg, line_search
from f3 import *
from progress.bar import Bar


np.seterr(all='raise')

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
    if len(r.shape) == 1:
        r = r.reshape((-1, 3))
    R2 = pdist(r, metric='sqeuclidean')
    V = np.sum(A/R2**6 - B/R2**3)
    return V


def gradient_one_atom(r_other, x0, A=A, B=B):
    R2 = cdist(r_other, x0.reshape((1, 3)), metric='sqeuclidean')
    frac = - 12*A/R2**7 + 6*B/R2**4
    coord_dists = x0 - r_other
    grad = np.sum(coord_dists*frac, axis=0)
    return grad


def gradient_total(r, A=A, B=B, normalize=False):
    grad = np.zeros(r.shape)
    for n, atom in enumerate(r):
        r_other = r[np.arange(r.shape[0]) != n]
        grad[n] = gradient_one_atom(r_other, atom, A, B)

    if normalize:
        grad = grad/abs(np.max(grad))
    return grad


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
        try:
            grad = grad/(abs(np.max(grad)))
        except FloatingPointError as e:
            print(np.max(grad))
            print(grad)
            raise e
    return grad




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
    alpha_max = 1e15
    x = conjugate_gradient(potential_total, data, g=get_gradient,
                           alpha_max=alpha_max,
                           max_iter=9, epsilon=1e-6)
    # N = len(x)
    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

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
    x = x[-1]
    grad = get_gradient(potential_total, x, normalize=False)
    print(grad)


    # ax.scatter(x[0], x[1], x[2])
    # plt.show()


def scipy_solution():
    data = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    result = fmin_cg(potential_total, data.flatten())
    x = result.reshape((-1, 3)).T
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.scatter(x[0], x[1], x[2])
    plt.show()


def main():
    q3()


if __name__ == "__main__":
    main()
