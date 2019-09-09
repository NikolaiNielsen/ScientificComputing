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
    print(f"Condition number for matrix with omega={omega}")
    print(f'{cond:4e}')


def householder_test():
    A = np.array([[1., 0, 0],
                  [0., 1, 0],
                  [0., 0, 1],
                  [-1., 1, 0],
                  [-1., 0, 1],
                  [0., -1, 1]])
    b = np.array((1237, 1941, 2417, 711, 1177, 475))
    Q, R = householder_QR(A)

if __name__ == "__main__":
    householder_test()
