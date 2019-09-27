import numpy as np
import matplotlib.pyplot as plt
from f2 import *
from examplematrices import *
from chladni_show import show_nodes, show_waves, \
    show_all_wavefunction_nodes
K = np.load('Chladni-Kmat.npy')
mats = [A1, A2, A3, A4, A5, A6]
eigenvals = [eigvals1, eigvals2, eigvals3, eigvals4, eigvals5, eigvals6]


def a():
    print("a2: Centers and radii of matrix K")
    centers, radii = gershgorin(K)
    for c, r in zip(centers, radii):
        print(f"c: {c:.4f} - r: {r:.4f}")
    print()


def b():
    np.random.seed(42)
    print("b3: Eigenvectors of example matrices")
    for n, (A, eigs) in enumerate(zip(mats, eigenvals)):
        x, k = power_iterate(A)
        approx = rayleigh_qt(A, x)
        res = np.sqrt(np.sum((A@x - approx*x)**2))
        print(f'A{n+1} converged after {k} iterations')
        # print(f'Eigenvector: {x}')
        print(f"Eigenvalue: {approx}")
        print(f'Residual: {res}')
        print()
    print()

    print("b4: largest eigenvalue of K")
    x, k = power_iterate(K, epsilon=1e-9, max_iter=50)
    eig = rayleigh_qt(K, x)
    res = np.sqrt(np.sum((K@x - eig*x)**2))
    print(f'eigenvalue: {eig}')
    print(f'Residual: {res}')
    print(f'iterations: {k}')
    print()


def c():
    np.random.seed(42)
    print('c2: Eigenvalues of example matrices')
    for n, (A, eigs) in enumerate(zip(mats, eigenvals)):
        x, k = rayleigh_iterate(A)
        approx = rayleigh_qt(A, x)
        res = np.sqrt(np.sum((A@x - approx*x)**2))
        print(f'A{n+1} converged after {k} iterations')
        print(f"Eigenvalue: {approx}")
        print(f'Residual: {res}')
        print()
    print()


def d(epsilon=1e-6):
    print(f"D: more eigenvalues! epsilon={epsilon:.2e}")
    centers, radii = gershgorin(K)
    lower = centers - radii
    higher = centers + radii
    eigs = []
    eigvs = []

    for shifts in zip(lower, higher, centers):
        for shift in shifts:
            x, k = rayleigh_iterate(K, shift=shift, epsilon=epsilon)
            if x is not None:
                eigs.append(rayleigh_qt(K, x))
                eigvs.append(x)

    eigs = np.array(eigs)
    eigvs = np.array(eigvs)

    unique = find_unique(eigs, eigvs)[::-1]
    print("Unique eigenvalues of K:")
    for u in unique:
        print(f'{u[0]:.4f}')

    Lambda = np.zeros(15)
    U = np.zeros((15, 15))
    for n, u in enumerate(unique):
        Lambda[n] = u[0]
        # Remember to normalize according to 2-norm
        U[:, n] = u[1]/np.sqrt(np.sum(u[1]**2))
    print()
    Lambda_real = np.linalg.inv(U) @ K @ U
    res = Lambda_real - np.diag(Lambda)
    Norm = np.max(np.sum(np.abs(res), axis=1))

    print("Max norm of the residual of Lambda:")
    print(Norm)

    # We only want to show nodes if we are using a better than default epsilon
    if epsilon < 1e-6:
        show_all_wavefunction_nodes(U, Lambda)


def main():
    a()
    b()
    c()
    d()
    d(epsilon=1e-10)

if __name__ == "__main__":
    main()
