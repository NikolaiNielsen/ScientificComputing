import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist
from f3 import *


EPSILON = 0.997
SIGMA = 3.401
A = 4*EPSILON*SIGMA**12
B = 4*EPSILON*SIGMA**6


def potential(R2):
    """
    Calculates the total interatomic potential (assuming Lennard Jones
    potential). Takes all interatomic distances squared as input.
    """
    V = np.sum(A/R2**6 - B/R2**3, axis=0)
    return V


def potential_total(r):
    """
    Calculates the total interatomic potential (assuming Lennard Jones
    potential). Takes all absolute coordinates as input
    """
    R2 = pdist(r, metric='sqeuclidean')
    V = np.sum(A/R2**6 - B/R2**3)
    return V


def get_gradient(r, h=1e-4, normalize=True):
    """
    Calculates the gradient of V_total
    """
    N = r.size//3
    grad = np.zeros(r.shape)
    for i in range(N):
        remaining_atoms = r[np.arange(N) != i]
        x, y, z = r[i]
        variations = np.array([[x+h, y, z],
                               [x-h, y, z],
                               [x, y+h, z],
                               [x, y-h, z],
                               [x, y, z+h],
                               [x, y, z-h]])
        variated_distances = cdist(remaining_atoms, variations,
                                   metric='sqeuclidean')
        varied_potentials = potential(variated_distances)
        dx = (varied_potentials[0] - varied_potentials[1])/(2*h)
        dy = (varied_potentials[2] - varied_potentials[3])/(2*h)
        dz = (varied_potentials[4] - varied_potentials[5])/(2*h)
        grad[i] = [dx, dy, dz]

    if normalize:
        grad = grad/abs(np.max(grad))
    return grad


def test():
    data = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    print(potential_total(data))
    grad = get_gradient(data)
    print(grad)


if __name__ == "__main__":
    test()
