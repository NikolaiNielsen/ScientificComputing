import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import fmin_cg
from f3 import *
from progress.bar import Bar
from p3 import get_gradient, gradient_total

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

    AnalyticGrad = np.array([[-6, -6, 0],
                             [183/32, 9/32, 0],
                             [9/32, 183/32, 0]])

    AnalyticalPot = -7/64

    numericalGrad = get_gradient(pot, r, normalize=False)
    analGrad = gradient_total(r, A=1, B=1, normalize=False)
    res = AnalyticGrad - analGrad

    # print(f'Gradient: {numericalGrad}')
    print('Testing potential calculation and gradient')
    print(analGrad)
    print(f'Gradient residual max norm: {np.max(np.sum(res, axis=1))}')

    # print(f'potential: {pot(r)}')
    print(f'Potential residual:  {pot(r)-AnalyticalPot}')


def test_gradient():
    data = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    grad = gradient_total(data)
    s = -grad
    x = data + s
    c = ['b']*20 + ['r']*20
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.scatter(data.T[0], data.T[1], data.T[2], c=c)
    ax.quiver(data.T[0], data.T[1], data.T[2],
              s.T[0], s.T[1], s.T[2], length=0.02)
    ax.scatter(x.T[0], x.T[1], x.T[2], c=c)
    plt.show()


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
    alpha_max = 1
    x, s = conjugate_gradient(potential_total, data, g=gradient_total,
                              alpha_max=alpha_max,
                              max_iter=9, epsilon=1e-6, return_s=True)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), nrows=3, ncols=3)
    ax = ax.flatten()
    for n, (Ax, X, S) in enumerate(zip(ax, x, s)):
        Xs = X.T
        ss = S.T
        grad = -gradient_total(Xs.T).T
        Ax.scatter(Xs[0], Xs[1], Xs[2])
        Ax.quiver(Xs[0], Xs[1], Xs[2], grad[0], grad[1], grad[2], length=0.01)
        Ax.set_title(f'{n}')
    fig.suptitle(r'First iterations. $\alpha_{max}= $' + str(alpha_max))
    fig.tight_layout()
    return fig, ax


def test_second_step():
    def evaluator(alpha, args):
        x0, s = args
        return x0 + alpha*s

    c = ['b']*20 + ['r']*20
    x0 = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    s = -gradient_total(x0)
    alpha = gss(potential_total, 0, 1, [evaluator, x0, s])
    x1 = x0 + alpha*s
    s2 = -gradient_total(x1)
    beta = s2.flatten().dot(s2.flatten()) / s.flatten().dot(s.flatten())
    s3 = s2 + beta*s

    fig, (ax1, ax2) = plt.subplots(subplot_kw=dict(projection='3d'), ncols=2)
    ax1.scatter(x1.T[0], x1.T[1], x1.T[2], c=c)
    ax1.quiver(x1.T[0], x1.T[1], x1.T[2], s2.T[0], s2.T[1], s2.T[2],
               length=0.01)

    alpha = gss(potential_total, 0, 1, [evaluator, x1, s2])
    print(alpha)
    x2 = x1 + alpha*s2
    s3 = -gradient_total(x2)
    ax2.scatter(x2.T[0], x2.T[1], x2.T[2], c=c)
    ax2.quiver(x2.T[0], x2.T[1], x2.T[2], s3.T[0], s3.T[1], s3.T[2],
               length=0.01)

    plt.show()


def show_line_search(Norm=True):
    data = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    g = get_gradient(potential_total, data, normalize=Norm)
    print(np.max(g))
    s = -g
    N = 10000
    n = 14
    alphas = np.logspace(1, 14, N)

    alphas2 = np.linspace(7e12, 1.5e13, N)
    f = np.zeros(N)
    f2 = np.zeros(N)
    for n, (alpha, alpha2) in enumerate(zip(alphas, alphas2)):
        f[n] = potential_total(data + alpha*s)
        f2[n] = potential_total(data + alpha2*s)

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(alphas, f)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$V_{tot}$')
    ax1.set_title('First linesearch')

    ax2.plot(alphas2, f2)
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$V_{tot}$')
    ax2.set_title('First linesearch')

    fig.tight_layout()
    return fig, (ax1, ax2)


def main():
    test_gss()
    test_potential()
    fig, ax = show_line_search()
    fig2, ax2 = show_first_iterations()
    plt.show()


if __name__ == "__main__":
    fig2, ax2 = show_first_iterations()
    plt.show()
