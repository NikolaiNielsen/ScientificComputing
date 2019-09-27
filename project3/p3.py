import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show3d(data):
    data.shape = 3, int(len(data) / 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0], data[1], data[2])
    data.shape = data.size
    return fig, ax


EPSILON = 0.997
SIGMA = 3.401
A = 4*EPSILON*SIGMA**12
B = 4*EPSILON*SIGMA**6


def potential(r, r0=0):
    R = r - r0
    V = A/R**12 - B/R**6
    return V


def newton_raphson(f, x0, h=5e-2, max_iter=50, epsilon=1e-3):
    x = [x0]
    for i in range(max_iter):
        fx = f(x[-1])
        fminus = f(x[-1]-h)
        fplus = f(x[-1]+h)
        diff = (fplus-fminus)/(2*h)
        x_new = x[-1] - fx/diff
        x.append(x_new)
        res = abs((x[-1] - x[-2])/x[-2])
        if res <= epsilon:
            break

    return np.array(x)


def bisection(f, a, b, epsilon=1e-6):
    a, b = min(a, b), max(a, b)
    while b-a > epsilon:
        m = a + (b-a)/2
        fa = f(a)
        fm = f(m)
        sfa = fa/abs(fa)
        fsm = fm/abs(fm)
        if sfa == fsm:
            a = m
        else:
            b = m
    return m


def secant(f, x0, x1, max_iter=100, epsilon=1e-6):
    x = [x0, x1]
    fx = [f(x0), f(x1)]

    for i in range(max_iter):
        x_new = x[-1] - fx[-1]*(x[-1] - x[-2])/(fx[-1] - fx[-2])
        fx.append(f(x_new))
        x.append(x_new)
        if abs((x[-1] - x[-2])/x[-2]) < epsilon:
            break
    return x


def inverse_quadratic(f, a, b, c, max_iter=100, epsilon=1e-6):
    # Sort a, b and c
    # a, b, c = sorted([a, b, c])
    guesses = []
    for i in range(max_iter):
        fa = f(a)
        fb = f(b)
        fc = f(c)

        u = fb/fc
        v = fb/fa
        w = fa/fc
        p = v * (w * (u-w) * (c-b) - (1-u) * (b-a))
        q = (w-1) * (u-1) * (v-1)

        if abs(p/q) < epsilon:
            break

        a, b, c = b, b + p/q, a
        guesses.append(b)

    return guesses


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


def main():
    q1()


if __name__ == "__main__":
    main()
