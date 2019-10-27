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
    """
    Basic experiments
    """
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
    """
    Simulate the effects of transfusions.
    """
    # Standard parameters and initial conditions
    params = [10., 5, 5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 5, 100, 100]
    x0 = [0.01, 0, 0, 0]
    params = np.array(params)
    x0 = np.array(x0)

    # Preparing the array of parameters
    N_e = 101
    e = np.linspace(0, 1, N_e)
    ones = np.ones(N_e)
    # We keep all parameters except e1 the same
    params = np.outer(ones, params)
    params[:, 8] = e
    x0 = np.outer(ones, x0)

    x_, t = sim(f, x0, params)
    # Save only the heterosexual male arrays
    z = x_[:, -1, :]
    tt, ee = np.meshgrid(t, e)
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


def deaths1():
    """
    Analyzing death rates, one by one.
    """

    # Setting up basic parameters
    N_r = 151
    params = [10., 5, 5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 5, 100, 100]
    x0 = [0.01, 0, 0, 0]
    params = np.array(params)
    x0 = np.array(x0)
    ones = np.ones(N_r)
    params2 = np.outer(ones, params)
    x0 = np.outer(ones, x0)

    # Death rates
    r1 = np.linspace(0, 3, num=N_r)
    r2 = np.linspace(0, 20, num=N_r)
    r3 = np.linspace(0, 60, num=N_r)
    r4 = np.linspace(0, 60, num=N_r)

    # Analytical expressions of equilibrium values
    def calc_r1(r1, params):
        a1, a2, p1, p2 = params[[0, 1, -4, -3]]
        a = -a1
        b = a1*p1-a2*p2-r1
        c = a2*p2*p1
        x1 = (-b-np.sqrt(b*b-4*a*c))/(2*a)
        return x1

    def calc_r2(r2, params):
        b1, b2, b3, p1, p2, q = params[[2, 3, 4, -4, -3, -2]]
        a = -b2
        b = b2*p2 - b1*p1 - b3*q - r2
        c = b1*p1*p2 + b2*q*p2
        x2 = (-b-np.sqrt(b*b-4*a*c))/(2*a)
        return x2

    def calc_r3(r3, params):
        c1, c2, p2, q, r = params[[5, 6, -3, -2, -1]]
        y = (c1*p2*q + c2*r*q)/(c1*p2+c2*r+r3)
        return y

    def calc_r4(r4, params):
        d1, e, p1, q, r = params[[7, 8, -4, -2, -1]]
        z = (d1*q*r + e*p1*r)/(d1*q+e*p1+r4)
        return z

    # Little helper function for getting the relevant simulation results.
    def get_results(death_rate, variable_num, analytical_func, N_t=500, x0=x0,
                    params=params2):
        # Create 2D matrix with only varying death rate for relevant population
        params = params.copy()
        params[:, -4+variable_num] = death_rate

        # Get analytical and simulated results.
        analytic = analytical_func(death_rate, params[0])
        x, t = sim(f, x0, params, N=N_t)

        # Keep only relevant population, calculate residual and relative error
        simulation = x[:, variable_num, -1]
        res = simulation-analytic
        rel = res/analytic
        return simulation, analytic, rel

    sim_x1, analytical_x1, rel_x1 = get_results(r1, -4, calc_r1)
    sim_x2, analytical_x2, rel_x2 = get_results(r2, -3, calc_r2)
    sim_y, analytical_y, rel_y = get_results(r3, -2, calc_r3)
    sim_z, analytical_z, rel_z = get_results(r4, -1, calc_r4)

    fig, ax = plt.subplots(ncols=2, nrows=4, figsize=(8, 12))
    ax = ax.flatten()

    ax[0].plot(r1, analytical_x1, label='Theory')
    ax[0].plot(r1, sim_x1, label='RK4')
    ax[0].legend()
    ax[0].set_xlabel('Death rate $r_1$')
    ax[0].set_ylabel('Equilibrium value of $x_1$')

    ax[1].plot(r1, rel_x1)
    ax[1].set_xlabel('Death rate $r_1$')
    ax[1].set_ylabel('Relative error')

    ax[2].plot(r2, analytical_x2, label='Theory')
    ax[2].plot(r2, sim_x2, label='RK4')
    ax[2].legend()
    ax[2].set_xlabel('Death rate $r_2$')
    ax[2].set_ylabel('Equilibrium value of $x_2$')

    ax[3].plot(r2, rel_x2)
    ax[3].set_xlabel('Death rate $r_2$')
    ax[3].set_ylabel('Relative error')

    ax[4].plot(r3, analytical_y, label='Theory')
    ax[4].plot(r3, sim_y, label='RK4')
    ax[4].legend()
    ax[4].set_xlabel('Death rate $r_3$')
    ax[4].set_ylabel('Equilibrium value of $y$')

    ax[5].plot(r3, rel_y)
    ax[5].set_xlabel('Death rate $r_3$')
    ax[5].set_ylabel('Relative error')

    ax[6].plot(r4, analytical_z, label='Theory')
    ax[6].plot(r4, sim_z, label='RK4')
    ax[6].legend()
    ax[6].set_xlabel('Death rate $r_4$')
    ax[6].set_ylabel('Equilibrium value of $z$')

    ax[7].plot(r4, rel_z)
    ax[7].set_xlabel('Death rate $r_4$')
    ax[7].set_ylabel('Relative error')
    fig.tight_layout()
    fig.savefig('Deaths.pdf')


def deaths2():
    """
    Simulate the equilibrium populations with random death rates
    """
    # Standard parameters
    N_d = 4
    params = [10., 5, 5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 5, 100, 100]
    x0 = [0.01, 0, 0, 0]
    params = np.array(params)
    x0 = np.array(x0)
    ones = np.ones(N_d)
    params2 = np.outer(ones, params)
    x0 = np.outer(ones, x0)

    # Include death rates
    death_rates = np.random.uniform(low=0, high=[3, 20, 60, 60],
                                    size=(N_d, 4))
    params2[:, 9:13] = death_rates

    # Solve for equilibrium populations iteratively
    x0_eq = [4, 4, 90, 90]
    xs_eq = []
    ks = []
    for param in params2:
        x_eq, k = multivariate_newton(f, jacobian, x0_eq, param, return_k=True)
        xs_eq.append(x_eq)
        ks.append(k)
    x_eq = np.array(xs_eq).squeeze()
    k = np.array(ks)

    # Simulate and compare results
    x_sim, t = sim(f, x0, params2, N=500)
    x_sim_eq = x_sim[:, :, -1]
    res = (x_sim_eq - x_eq)
    rel = (res/x_eq)
    # Format for .tex table
    for rel_row, death_row, sim_eq, an_eq in zip(rel, death_rates,
                                                 x_sim_eq, x_eq):
        death_parts = [f'{i:.3f} & ' for i in death_row]
        s1 = 'Death Rate & ' + "".join(death_parts)[:-2] + r" \\"

        an_parts = [f'{i:.3f} & ' for i in an_eq]
        s2 = 'Analytical & ' + "".join(an_parts)[:-2] + r" \\"

        sim_parts = [f'{i:.3f} & ' for i in sim_eq]
        s3 = 'Simulated & ' + "".join(sim_parts)[:-2] + r" \\"

        rel_parts = [f'{i:.3e} & ' for i in rel_row]
        s4 = 'Relative error & ' + "".join(rel_parts)[:-2] + r" \\"
        print(s1)
        print(s2)
        print(s3)
        print(s4)
        print()
    # print(rel)


def main():
    deaths2()


if __name__ == "__main__":
    main()
