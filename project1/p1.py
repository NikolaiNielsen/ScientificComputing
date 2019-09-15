import numpy as np
from watermatrices import Amat, Bmat, yvec
from functions import *
import matplotlib.pyplot as plt

E = np.vstack((np.hstack((Amat, Bmat)), np.hstack((Bmat, Amat))))
diag = np.array((*[1]*7, *[-1]*7))
S = np.diag(diag)
z = np.array([*yvec, *-yvec])


def a2():
    omega = np.array((1.300, 1.607, 3.000))
    z = 0.5e-8
    print("\nAnswer a2:")
    for o in omega:
        M = E - o*S
        cond = calc_cond(M)
        print(f"Condition number for omega={o}")
        print(cond)
        print(np.floor(-np.log10(cond*z)))


def b1():
    omega = np.array((1.300, 1.607, 3.000))
    print("\nAnswer b1:")
    for o in omega:
        e = forward_error_bound(E, S, o)
        print(f"Error bound for omega={o}:")
        print(f"{e:.4e}")
        print(np.floor(-np.log10(e)))


def c():
    A = np.array([[2, 1, 1], [4, 1, 4], [-6, -5, 3]])
    b = np.array([4, 11, 4])
    x_np = np.linalg.solve(A, b)
    x_me = linsolve(A, b)
    print("\nAnswer c:")
    print("Solving linear system Ax=b")
    print(x_me)
    res = np.linalg.norm(x_me-x_np, ord=2)
    print(res)


def d1():
    omega = np.array([1.300, 1.607, 3.000])
    domega = 5e-4

    print("\nAnswer d1:")
    for o in omega:
        alpha = solve_alpha(o, E, S, z)
        print(f"alpha(omega) for omega={o}")
        print(alpha)
        upper = solve_alpha(o+domega, E, S, z)
        lower = solve_alpha(o-domega, E, S, z)
        print(f'Upper bound')
        print(upper)
        print(f'Lower bound')
        print(lower)
        print(max(abs((alpha-upper)/alpha), abs((alpha-lower)/alpha)))


def e1():
    omega = np.linspace(1.2, 4, 1000)
    alpha = np.zeros(omega.shape)

    for n in range(omega.size):
        alpha[n] = solve_alpha(omega[n], E, S, z)

    fig, ax = plt.subplots()
    ax.plot(omega, alpha)
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$\alpha(\omega)$')
    ax.set_title('Polarizability of water, as a function of the frequency')
    fig.savefig('polarizability.pdf')


def e2():
    omega = 1.60686978
    M = E-omega*S
    cond = calc_cond(M)
    print("\nAnswer e2:")
    print(f"Condition number for matrix with omega={omega}: {cond:4e}")


def f():
    A = np.array([[1., 0, 0],
                  [0., 1, 0],
                  [0., 0, 1],
                  [-1., 1, 0],
                  [-1., 0, 1],
                  [0., -1, 1]])
    b = np.array((1237, 1941, 2417, 711, 1177, 475))
    Q, R = householder_QR(A, inline=False)
    id_ = Q.T @ Q
    x = least_squares(A, b)

    print('f:')
    print(f'np.islcose(Q^T Q,I)={np.isclose(id_, np.identity(b.size)).all()}')
    print(f'np.isclose(A,QR)={np.isclose(A, Q@R).all()}')
    print(f'linear least square fit: x={x}')


def g():
    omega_p = 1.6
    omega = np.linspace(1.2, 4, 1000)
    omega = omega[omega < omega_p]
    alpha = np.zeros(omega.shape)
    for n in range(omega.size):
        alpha[n] = solve_alpha(omega[n], E, S, z)

    x1, P1 = least_squares_P(omega, alpha, 4)
    x2, P2 = least_squares_P(omega, alpha, 6)
    print('g: parameters:')
    print(x1)
    print(x2)

    rel1 = np.abs((P1-alpha)/alpha)
    rel2 = np.abs((P2-alpha)/alpha)

    fig2, (ax2, ax3) = plt.subplots(ncols=2, figsize=(8, 4))
    ax2.plot(omega, rel1, label='n=4')
    ax2.plot(omega, rel2, label='n=6')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\omega$')
    ax2.set_ylabel(
        r'$\log_{10} |(P(\omega) - \alpha(\omega))/\alpha(\omega)|$')
    ax2.legend()

    ax3.plot(omega, np.floor(-np.log10(rel1)), label='n=4')
    ax3.plot(omega, np.floor(-np.log10(rel2)), label='n=6')
    ax3.legend()
    ax3.set_xlabel(r'$\omega$')
    ax3.set_ylabel(r'Number of significant digits of $P(\omega)$')
    fig2.tight_layout()
    fig2.savefig('g34.pdf')


def h():
    omega = np.linspace(1.2, 4, 1000)
    alpha = np.zeros(omega.shape)
    for n in range(omega.size):
        alpha[n] = solve_alpha(omega[n], E, S, z)

    params1, Q1 = least_squares_Q(omega, alpha, 2)
    params2, Q2 = least_squares_Q(omega, alpha, 4)

    a1 = params1[:3]
    b1 = params1[3:]
    a2 = params2[:5]
    b2 = params2[5:]
    print("h: parameters")
    print(a1, b1)
    print(a2, b2)

    rel1 = np.abs((Q1-alpha)/alpha)
    rel2 = np.abs((Q2-alpha)/alpha)
    fig2, (ax2, ax3) = plt.subplots(ncols=2, figsize=(8, 4))
    ax2.plot(omega, rel1, label='n=2')
    ax2.plot(omega, rel2, label='n=4')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\omega$')
    ax2.set_ylabel(
        r'$\log_{10} |(Q(\omega) - \alpha(\omega))/\alpha(\omega)|$')
    ax2.legend()

    ax3.plot(omega, np.floor(-np.log10(rel1)), label='n=2')
    ax3.plot(omega, np.floor(-np.log10(rel2)), label='n=4')
    ax3.legend()
    ax3.set_xlabel(r'$\omega$')
    ax3.set_ylabel(r'Number of significant digits of $Q(\omega)$')
    fig2.tight_layout()
    fig2.savefig('h.pdf')


def main():
    a2()
    b1()
    c()
    d1()
    e1()
    e2()
    f()
    g()
    h()

if __name__ == "__main__":
    main()
