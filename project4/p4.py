import numpy as np
import matplotlib.pyplot as plt
from f4 import *


def jacobian(x, params):
    """
    Jacobian for Differential equation for HIV transmittance.
    simulates for n different set of parameters

    Inputs:
    x - [4,]
    params - [17,]
    """
    a1, a2, b1, b2, b3, c1, c2, d1, e, r1, r2, r3, r4, p1, p2, q, r = params.T
    x1, x2, y, z = x.T
    jac = np.zeros(4, 4)
    jac[0, 0] = a1*(p1-2*x1) - a2*x2 - r1
    jac[0, 1] = a2*(p1-x1)
    jac[1, 0] = b1*(p2-x2)
    jac[1, 1] = -b1*x1 + b2*(p2-2*x2) - b3*y - r2
    jac[1, 2] = b3*(p2-x2)
    jac[2, 1] = c1*(q-y)
    jac[2, 2] = -c1*x2 - c2*z - r3
    jac[2, 3] = c2*(q-y)
    jac[3, 0] = e*(r-z)
    jac[3, 2] = d1*(r-z)
    jac[3, 3] = -d1*y - e*x1 - r3

    return jac


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