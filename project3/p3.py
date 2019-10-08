import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import fmin_cg, line_search
from f3 import *
import timeit
import time


np.seterr(all='raise')

EPSILON = 0.997
SIGMA = 3.401
A = 4*EPSILON*SIGMA**12
B = 4*EPSILON*SIGMA**6


def potential1D(r, r0=0):
    V = A/(r-r0)**12 - B/(r-r0)**6
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
    """
    Calculates 3 parameters of the gradient of VLJ (ie, one atom) analytically
    """
    R2 = cdist(r_other, x0.reshape((1, 3)), metric='sqeuclidean')
    frac = - 12*A/R2**7 + 6*B/R2**4
    coord_dists = r_other - x0
    grad = np.sum(coord_dists*frac, axis=0)
    return grad


def gradient_total(r, A=A, B=B, normalize=True):
    """
    Calculate the total gradient of the Lennard Jones potential.
    """
    grad = np.zeros(r.shape)
    for n, atom in enumerate(r):
        r_other = r[np.arange(r.shape[0]) != n]
        grad[n] = gradient_one_atom(r_other, atom, A, B)
    
    # Since we normalize we must account for this in calculating beta.
    max_ = abs(np.max(grad))
    if normalize:
        grad = grad/max_
    return grad, max_


def gradient_total2(r, A=A, B=B, normalize=True):
    r = r.reshape((-1, 3))
    grad = np.zeros(r.shape)
    for n, atom in enumerate(r):
        r_other = r[np.arange(r.shape[0]) != n]
        grad[n] = gradient_one_atom(r_other, atom, A, B)

    if normalize:
        grad = grad/abs(np.max(grad))
    return grad.flatten()


def get_gradient(f, r, h=1e-4, normalize=True):
    """
    Calculates the gradient of V_total numerically with a central difference
    approximation.
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
    p1 = potential1D(r1, r0)
    fig, (ax, ax2) = plt.subplots(nrows=2)
    ax.plot(r1, p1, label='V > 0')
    ax.plot(r1[p1 < 0], p1[p1 < 0], label='V < 0')
    r2 = np.linspace(3, 5, 100)
    p2 = potential1D(r2, r0)
    ax2.plot(r2, p2, label='V > 0')
    ax2.plot(r2[p2 < 0], p2[p2 < 0], label='V < 0')
    ax.set_xlabel('Distance $r$')
    ax.set_ylabel('Potential $V$')
    ax2.set_xlabel('Distance $r$')
    ax2.set_ylabel('Potential $V$')
    ax.legend()
    ax2.legend()
    ax.set_title('12-6 Lennard-Jones potential between two atoms')
    fig.tight_layout()

    setup = 'from p3 import potential_total\nimport numpy as np\n'
    setup += "data = np.genfromtxt('Ar-lines.csv', delimiter=' ')"
    N = 1000
    exec_time = timeit.timeit("potential_total(data)", setup=setup, number=N)/N
    print(f"Avg calculation time of total potential: {exec_time:.4e} s")
    return fig, ax, ax2


def q2():
    x1 = bisection(potential1D, 2, 4)
    x2 = newton_raphson(potential1D, x0=2)[-1]
    x3 = secant(potential1D, 1.65, 2)[-1]
    x4 = inverse_quadratic(potential1D, 1, 2, 3)[-1]

    s = "from f3 import newton_raphson, bisection, secant, inverse_quadratic"
    s = s + "\nfrom p3 import potential1D"
    N = 10000
    bisection_time = timeit.timeit("bisection(potential1D, 2, 4)",
                                   setup=s, number=N)/N
    newton_raphson_time = timeit.timeit("newton_raphson(potential1D, x0=2)",
                                        setup=s, number=N)/N
    secant_time = timeit.timeit("secant(potential1D, 1.65, 2)",
                                setup=s, number=N)/N
    quadratic_time = timeit.timeit("inverse_quadratic(potential1D, 1, 2, 3)",
                                   setup=s, number=N)/N

    print("Roots of 12-6 Lennard-Jones potential found:\n")
    print("Bisection Method:")
    print(f"Root: {x1:.5f}")
    print(f"Avg time: {bisection_time:.3e}\n")

    print("Newton-Raphson Method with numerical derivative:")
    print(f"Root: {x2:.5f}")
    print(f"Avg time: {newton_raphson_time:.3e}\n")

    print("Secant Method:")
    print(f"Root: {x3:.5f}")
    print(f"Avg time: {secant_time:.3e}\n")

    print("Inverse Quadratic Method:")
    print(f"Root: {x4:.5f}")
    print(f"Avg time: {quadratic_time:.3e}")


def q3():
    data = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    alpha_max = 2

    t0 = time.time()
    x = conjugate_gradient(potential_total, data, g=gradient_total,
                           g2=gradient_total2,
                           alpha_max=alpha_max,
                           max_iter=2000, epsilon=1e-7)
    t1 = time.time()
    totT = (t1-t0)
    energies = [potential_total(r) for r in x]

    print(f"Minimum potential {energies[-1]:.3e} found in {totT:.3e} seconds.")

    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212)
    last = x[-1].T
    ax1.scatter(last[0], last[1], last[2])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax2.plot(energies)
    ax2.set_yscale('log')
    ax2.set_xlabel('Iteration number')
    ax2.set_ylabel('$V_{tot}$')
    fig.savefig("q3fig.pdf")
    # plt.show()


def q4():
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


def scipy_solution():
    data = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    # alliters = np.load('alliter.npy')
    t0 = time.time()
    result, alliters = fmin_cg(potential_total, data.flatten(), retall=True)
    t1 = time.time()
    totT = t1-t0
    x = result.reshape((40, 3))
    pot = potential_total(x)
    print(f"Minimum potential {pot:.3e} found in {totT:.3e} seconds.")


def main():
    fig, ax1, ax2 = q1()
    fig.savefig('q1fig.pdf')
    q2()
    q3()
    scipy_solution()
    q4()


if __name__ == "__main__":
    main()
