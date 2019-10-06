import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import fmin_cg, line_search
from f3 import *
from progress.bar import Bar
import timeit

EPSILON = 0.997
SIGMA = 3.401
A = 4*EPSILON*SIGMA**12
B = 4*EPSILON*SIGMA**6


def potential1D(r, r0=0):
    V = A/(r-r0)**12 - B/(r-r0)**6
    return V


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


def neighbour(x, stepsize=1):
    choice = np.random.randint(x.size)
    x[choice] += np.random.uniform(-stepsize/2, stepsize/2)
    return x


def neighbour_gauss(x, stepsize=1):
    choice = np.random.randint(x.size)
    x[choice] += np.random.normal(scale=stepsize)
    return x


def acceptance_probability(old, new, T):
    return np.exp((-new+old)/T)


def anneal(f, x0, neighbour=neighbour, NT=100, Nf=100, tol=1e-3):
    x_last = x0
    cost_last = f(x_last)
    costs = [cost_last]
    T = 1.0
    T_min = 1e-5
    alpha = (T_min/T)**(1/NT)
    bar = Bar('Annealing', max=NT)
    while T > T_min:
        for _ in range(100):
            x_new = neighbour(x_last)
            cost_new = f(x_new)
            ap = acceptance_probability(cost_last, cost_new, T)
            if ap > np.random.uniform():
                x_last = x_new
                cost_last = cost_new
                costs.append(cost_last)
        T = T * alpha
        bar.next()
        if cost_last <= tol:
            break
    bar.finish()
    return x_last, costs


def costFunc(r):
    return potential_total(r)


def setup():
    np.random.seed(42)
    data = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    data = data.flatten()
    x, costs = anneal(costFunc, data, tol=-50, NT=10000,
                      neighbour=neighbour)
    x = x.reshape((-1, 3))
    costs = np.abs(np.array(costs))

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.scatter(x.T[0], x.T[1], x.T[2])

    fig2, ax2 = plt.subplots()
    ax2.plot(costs)
    ax2.set_yscale('log')
    print(costs[-1])
    plt.show()


def main():
    setup()


if __name__ == "__main__":
    main()
