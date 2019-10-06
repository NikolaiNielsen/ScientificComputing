import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import fmin_cg, line_search
from f3 import *
from progress.bar import Bar
import timeit
import time

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
    # Pick a random parameter, and add a random amount (uniform) centred around
    # 0, with size stepsize.
    choice = np.random.randint(x.size)
    x[choice] += np.random.uniform(-stepsize/2, stepsize/2)
    return x


def neighbour_gauss(x, stepsize=1):
    choice = np.random.randint(x.size)
    x[choice] += np.random.normal(scale=stepsize)
    return x


def anneal(f, x0, neighbour=neighbour, neigharg=1, NT=100, Nf=100, target=1e-3,
           minimize=True):

    if not minimize:
        # If we wanna maximize, we just change the sign of the cost function
        # and target.
        def neigh(x, arg):
            return -neighbour(x, arg)
        target = -target
    else:
        neigh = neighbour

    # Set up starting parameters
    x = x0
    cost = f(x)
    xs = [x]
    costs = [cost]

    # Initial temperature and final temperature
    T = 1.0
    T_min = 1e-5
    # Scaling factor for T, chosen to temperature is changed NT times
    alpha = (T_min/T)**(1/NT)

    while T > T_min:
        # For each temperature we test Nf neighbours.
        for _ in range(Nf):
            x_new = neigh(x, neigharg)
            cost_new = f(x_new)

            # if cost_new is better than cost, it is automatically accepted,
            # since ap >= 1
            ap = np.exp((cost-cost_new)/T)
            if ap > np.random.uniform():
                x = x_new
                cost = cost_new
                costs.append(cost)
                xs.append(x)

        # Decrease temperature
        T = T * alpha

        if cost <= target:
            break
    return xs, costs


def setup():
    np.random.seed(42)
    data = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    data = data.flatten()
    t0 = time.time()
    x, costs = anneal(potential_total, data, target=0, NT=10000,
                      neighbour=neighbour, neigharg=0.2)
    t1 = time.time()
    totT = t1-t0
    x = x[-1].reshape((-1, 3))
    costs = np.abs(np.array(costs))
    print(f"Minimum potential {costs[-1]:.3e} found in {totT:.3e} seconds.")

    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212)
    ax1.scatter(x.T[0], x.T[1], x.T[2])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax2.plot(costs)
    ax2.set_yscale('log')
    ax2.set_xlabel('Iteration number')
    ax2.set_ylabel('$V_{tot}$')
    fig.savefig("q4fig.pdf")


def main():
    setup()


if __name__ == "__main__":
    main()
