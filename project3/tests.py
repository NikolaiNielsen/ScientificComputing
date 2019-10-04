import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import fmin_cg
from f3 import *
from progress.bar import Bar
from p3 import get_gradient

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


def test_potential():
    def pot(r):
        return potential_total(r, 1, 1)

    r = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0.]])

    AnalyticGrad = np.array([[6, 6, 0],
                             [-183/32, -9/32, 0],
                             [-9/32, -183/32, 0]])

    AnalyticalPot = -7/64

    numericalGrad = get_gradient(pot, r, normalize=False)
    res = AnalyticGrad - numericalGrad

    # print(f'Gradient: {numericalGrad}')
    print('Testing potential calculation and gradient')
    print(f'Gradient residual max norm: {np.max(np.sum(res, axis=1))}')

    # print(f'potential: {pot(r)}')
    print(f'Potential residual:  {pot(r)-AnalyticalPot}')


def test_gss():
    def f(x):
        return 0.5 - x*np.exp(-x**2)

    analytical_minimum = np.sqrt(2)/2
    a, b = 0, 2
    x = gss(f, a, b)

    print('Testing GSS on 0.5-x*exp(-x*x)')
    print(f'minimum:  {x}')
    print(f'residual: {x-analytical_minimum}')


def show_first_iterations():
    data = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    alpha_max = 0.1
    x = conjugate_gradient(potential_total, data, g=get_gradient,
                           alpha_max=alpha_max,
                           max_iter=9, epsilon=1e-6)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), nrows=3, ncols=3)
    ax = ax.flatten()
    for n, (Ax, X) in enumerate(zip(ax, x)):
        Xs = X.T
        Ax.scatter(Xs[0], Xs[1], Xs[2])
        Ax.set_title(f'{n}')
    fig.suptitle(r'First iterations. $\alpha_{max}= $' + str(alpha_max))
    return fig, ax


def show_line_search(Norm=True):
    data = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    g = get_gradient(potential_total, data, normalize=Norm)
    s = -g
    N = 10000
    n = 14
    alphas = np.logspace(1, 14, N)
    f = np.zeros(N)
    for n, alpha in enumerate(alphas):
        f[n] = potential_total(data + alpha*s)

    fig, ax = plt.subplots()
    ax.plot(alphas, f)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$V_{tot}$')
    ax.set_title('First linesearch')
    return fig, ax


def main():
    test_gss()
    test_potential()
    fig, ax = show_line_search()
    fig2, ax2 = show_first_iterations()
    plt.show()


if __name__ == "__main__":
    main()
