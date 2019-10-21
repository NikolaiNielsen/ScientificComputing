import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    jac = np.zeros((4, 4))
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


def find_equilibrium():
    params = [10, 5, 5, 1, 1, 1, 1, 1, 0, 0.05, 0.05, 0.05, 0.05, 5, 5, 100,
              100]
    x0 = [4, 4, 90, 90]
    x, k = multivariate_newton(f, jacobian, x0, params, return_k=True)
    print(x, k)


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
    ax1.set_yscale('linear')
    fig2, ax2 = create_plot(x2, t)
    ax2.set_title('With blood-transfusion')
    ax2.set_yscale('linear')
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


def no_deaths_or_transfusions():
    params = [10, 5, 5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 5, 100, 100]
    x0 = [0.01, 0, 0, 0]
    params = np.array(params)
    x0 = np.array(x0)
    x_e, t_e = sim(f, x0, params, sim_method=forward_euler)
    x_rk, t_rk = sim(f, x0, params)

    x_e = x_e.squeeze()
    x_rk = x_rk.squeeze()

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    x1, x2, y, z = x_e
    ax1.plot(t_e, x1)
    ax1.plot(t_e, x2)
    ax1.plot(t_e, y)
    ax1.plot(t_e, z)
    ax1.legend(['Homosexual men', 'Bisexual men', 'Women', 'Heterosexual men'])
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Population with HIV')
    ax1.set_title('Spread of HIV in population')
    ax1.set_yscale('log')
    ax1.set_title('Forward Euler method')

    x1, x2, y, z = x_rk
    ax2.plot(t_rk, x1)
    ax2.plot(t_rk, x2)
    ax2.plot(t_rk, y)
    ax2.plot(t_rk, z)
    ax2.legend(['Homosexual men', 'Bisexual men', 'Women', 'Heterosexual men'])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Population with HIV')
    ax2.set_title('Spread of HIV in population')
    ax2.set_yscale('log')
    ax2.set_title('4th order Runge-Kutta method')

    print('No deaths or transfusions:')
    print('final values with Euler method:')
    print(x_e[:, -1])
    print('Final value with RK4 method:')
    print(x_rk[:, -1])

    fig.tight_layout()
    fig.savefig('no_death_or_transfusion.pdf')


def transfusions():
    N_e = 101
    e = np.linspace(0, 1, N_e)
    params = [10., 5, 5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 5, 100, 100]
    x0 = [0.01, 0, 0, 0]
    params = np.array(params)
    x0 = np.array(x0)
    sim_data = False

    if sim_data:
        z = []
        for i in range(N_e):
            params[8] = e[i]
            x_, t = sim(f, x0, params)
            x_ = x_.squeeze()
            z.append(x_[-1])
        z = np.array(z)
        np.save('transfusion_data', z)
    else:
        z = np.load('transfusion_data.npy')
        N_e, N_t = z.shape
        dt = 5e-4
        t = np.cumsum(np.zeros(N_t)+dt) - dt
        e = np.linspace(0, 1, N_e)

    tt, ee = np.meshgrid(t, e)

    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3.5))
    im = ax1.imshow(z, extent=[0, t[-1], 0, 1], aspect='auto', origin='lower',
                    cmap='coolwarm')
    ax1.contour(tt, ee, z, colors='xkcd:cement', levels=list(range(0, 101, 5)))
    fig.colorbar(im, cmap='coolwarm')
    fig.suptitle('Effect of blood transfusions on $z$')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Blood transfusion rate $e_1$')

    z_end = z[:, -1]
    ax2.plot(e, z_end)
    ax2.set_xlabel('Blood transfusion rate $e_1$')
    ax2.set_ylabel('Final infected population $z$')
    fig.tight_layout()
    fig.savefig('transfusions.pdf')


def main():
    no_deaths_or_transfusions()


if __name__ == "__main__":
    main()
