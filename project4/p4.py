import numpy as np
import matplotlib.pyplot as plt
from f4 import *


def create_plot(x, t):
    fig, ax = plt.subplots()
    x1, x2, y, z = x
    ax.plot(t, x1)
    ax.plot(t, x2)
    ax.plot(t, y)
    ax.plot(t, z)
    ax.legend(['Homosexual men', 'Bisexual men', 'Women', 'Heterosexual men'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Population with HIV')
    ax.set_title('Spread of HIV in population')
    ax.set_yscale('log')
    return fig, ax


def test_rk4():

    # Parameters:
    # a1, a2, b1, b2, b3, c1, c2, d1, e, r1, r2, r3, r4, p1, p2, q, r
    params = [10, 5, 5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 5, 100, 100]
    params2 = [10, 5, 5, 1, 1, 1, 1, 1, 0.002, 0, 0, 0, 0, 5, 5, 100, 100]
    params = np.array((params, params2))
    x0 = [0.01, 0, 0, 0]

    x0 = np.array((x0, x0))
    x, t = sim(f, x0, params, dt=5e-4, N=350)
    x = x.squeeze()
    x1 = x[0]
    x2 = x[1]
    fig1, ax1 = create_plot(x1, t)
    ax1.set_title('Without blood-transfusion')
    fig2, ax2 = create_plot(x2, t)
    ax2.set_title('With blood-transfusion')
    x3 = x2-x1
    fig3, ax3 = create_plot(x3, t)
    ax3.set_yscale('linear')
    ax3.set_title('Difference')
    plt.show()


def test_euler():

    # Parameters:
    # a1, a2, b1, b2, b3, c1, c2, d1, e, r1, r2, r3, r4, p1, p2, q, r
    params = [10, 5, 5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 5, 100, 100]
    params2 = [10, 5, 5, 1, 1, 1, 1, 1, 0, 0.05, 0.05, 0.05, 0.1, 5, 5,
               100, 100]
    params = np.array((params, params2))
    x0 = [0.01, 0, 0, 0]

    x0 = np.array((x0, x0))
    x, t = sim(f, x0, params, dt=5e-4, N=1000, sim_method=forward_euler)
    x = x.squeeze()
    x1 = x[0]
    x2 = x[1]
    fig1, ax1 = create_plot(x1, t)
    # ax1.set_yscale('linear')
    fig2, ax2 = create_plot(x2, t)
    # ax2.set_yscale('linear')
    plt.show()


def main():
    test_rk4()


if __name__ == "__main__":
    main()
