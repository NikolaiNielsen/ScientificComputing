import numpy as np
from f2 import *
from project2.examplematrices import *
K = np.load('project2/Chladni-Kmat.npy')
mats = [A1, A2, A3, A4, A5, A6]
eigenvals = [eigvals1, eigvals2, eigvals3, eigvals4, eigvals5, eigvals6]


def a():
    print("a2: Centers and radii of matrix K")
    centers, radii = gershgorin(K)
    for c, r in zip(centers, radii):
        print(c, r)
    print()


def b():
    print("b3: Eigenvectors of example matrices")
    for n, (A, eigs) in enumerate(zip(mats, eigenvals)):
        x, k = power_iterate(A)
        approx = rayleigh_qt(A, x)
        res = np.sqrt(np.sum((A@x - approx*x)**2))
        print(f'A{n} converged after {k} iterations')
        # print(f'Eigenvector: {x}')
        print(f"Eigenvalue: {approx}")
        print(f'Residual: {res}')
        print()
    print()

    print("b4: largest eigenvalue of K")
    x, k = power_iterate(K)
    print(f'eigenvalue: {rayleigh_qt(K, x)}')
    print()


def c():
    print('c2: Eigenvalues of example matrices')
    for n, (A, eigs) in enumerate(zip(mats, eigenvals)):
        x, k = rayleigh_iterate(A)
        approx = rayleigh_qt(A, x)
        res = np.sqrt(np.sum((A@x - approx*x)**2))
        print(f'A{n} converged after {k} iterations')
        # print(f'Eigenvector: {x}')
        print(f"Eigenvalue: {approx}")
        print(f'Residual: {res}')
        print()
    print()


def d():
    pass


def main():
    # a()
    # b()
    c()
    d()


if __name__ == "__main__":
    main()
