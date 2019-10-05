import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import fmin_cg, line_search
from f3 import *
from progress.bar import Bar
import timeit


np.seterr(all='raise')

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


def gradient_one_atom(r_other, x0, A=A, B=B):
    R2 = cdist(r_other, x0.reshape((1, 3)), metric='euclidean')
    frac = - 12*A/R2**14 + 6*B/R2**8
    coord_dists = r_other - x0
    grad = np.sum(coord_dists*frac, axis=0)
    return grad


def gradient_total(r, A=A, B=B, normalize=True):
    grad = np.zeros(r.shape)
    for n, atom in enumerate(r):
        r_other = r[np.arange(r.shape[0]) != n]
        grad[n] = gradient_one_atom(r_other, atom, A, B)

    if normalize:
        grad = grad/abs(np.max(grad))
    return grad


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
    N = 1000
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
    print(f"Root: {x1:.3f}")
    print(f"Avg time: {bisection_time:.3e}\n")

    print("Newton-Raphson Method with numerical derivative:")
    print(f"Root: {x2:.3f}")
    print(f"Avg time: {newton_raphson_time:.3e}\n")

    print("Secant Method:")
    print(f"Root: {x3:.3f}")
    print(f"Avg time: {secant_time:.3e}\n")

    print("Inverse Quadratic Method:")
    print(f"Root: {x4:.3f}")
    print(f"Avg time: {quadratic_time:.3e}")


def q3():
    data = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    alpha_max = 1e1
    x = conjugate_gradient(potential_total, data, g=gradient_total,
                           g2=gradient_total2,
                           alpha_max=alpha_max,
                           max_iter=2000, epsilon=1e-9)

    c = ['b']*20 + ['r']*20
    # for n in range(len(x)-1):
    #     res = x[n+1] - x[n]
    #     print(np.sum(np.sqrt(np.sum(res**2, axis=1))))
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    # writer = anim.FFMpegWriter(fps=1)
    # bar = Bar('Writing movie', max=len(x))
    # dpi = 200
    # outfile = 'movie.mp4'
    # with writer.saving(fig, outfile, dpi):
    #     for X in x:
    #         X = X.T
    #         ax.scatter(X[0], X[1], X[2], c=c)
    #         writer.grab_frame()
    #         ax.clear()
    #         bar.next()
    # bar.finish()

    last = x[-1].T
    ax.scatter(last[0], last[1], last[2])
    # ax.quiver(last[0], last[1], last[2], s[0], s[1], s[2], length=1)
    plt.show()


def scipy_solution():
    data = np.genfromtxt('Ar-lines.csv', delimiter=' ')
    alliters = np.load('alliter.npy')
    # result, alliters = fmin_cg(potential_total, data.flatten(), retall=True)
    # np.save('alliter', alliters)
    alliters = alliters.reshape((-1, 40, 3))
    first = alliters[1].T

    # x = result.reshape((-1, 3)).T
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    c = ['b']*20 + ['r']*20
    ax.scatter(first[0], first[1], first[2], c=c)
    plt.show()


def main():
    q3()


if __name__ == "__main__":
    main()
