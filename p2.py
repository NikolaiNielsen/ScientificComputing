import numpy as np
import matplotlib.pyplot as plt
from f2 import *
from project2.examplematrices import *
from project2.chladni_show import show_nodes, show_waves
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


def d():
    np.random.seed(42)
    print("d: As many eigenvalues of K as possible")
    centers, radii = gershgorin(K)
    low = centers - radii
    high = centers + radii
    print(low)
    eigs = []
    for n, (l, h, c) in enumerate(zip(low, high, centers)):
        x, k = rayleigh_iterate(K, shift=l)
        print()
        eigs.append(rayleigh_qt(K, x))
        x, k = rayleigh_iterate(K, shift=h)
        eigs.append(rayleigh_qt(K, x))

    eigs = np.array(eigs)
    # fig, ax = plt.subplots()
    # ax.scatter(np.arange(eigs.size), eigs)

    u = find_unique(eigs)
    print(u)


def d2():
    np.random.seed(42)
    print("d: As many eigenvalues of K as possible")
    eigs = []
    last_eig = 0
    for i in range(30):
        x, k = power_iterate(K, shift=last_eig)
        last_eig = rayleigh_qt(K, x)
        eigs.append(last_eig)
    # print(eigs)
    eigs = find_unique(np.array(eigs))
    new_eigs = []
    for eig in eigs:
        x, k = rayleigh_iterate(K, shift=eig)
        new_eigs.append(rayleigh_qt(K, x))
    # print(new_eigs)
    print(find_unique(np.array(new_eigs)))


def d3():
    np.random.seed(42)
    print("D3, more eigenvalues!")

    centers, radii = gershgorin(K)
    lower = centers - radii
    higher = centers + radii
    eigs = []
    eigvs = []
    N = 10
    for n, (l, h) in enumerate(zip(lower, higher)):
        shifts = np.linspace(l, h, N)
        for shift in shifts:
            x, k = rayleigh_iterate(K, shift=shift)
            eigs.append(rayleigh_qt(K, x))
            eigvs.append(x)

    eigs = np.array(eigs)
    eigvs = np.array(eigvs)
    # print(eigvs.shape)

    unique = find_unique(eigs, eigvs)[::-1]
    print(unique)


def d4():
    lambda_ = 151362.6666519405
    centers, radii = gershgorin(K)
    lower = centers - radii
    higher = centers + radii
    N = 10
    shifts = np.linspace(lower, higher, num=N, axis=1)
    shifts = np.sort(shifts.flatten())
    shifts = shifts[(-lambda_ <= shifts) * (shifts <= lambda_)]
    eigs = []
    eigvecs = []
    while len(shifts) > 0:
        shift = shifts[0]
        x, k = rayleigh_iterate(K, shift=shift)
        new_eig = rayleigh_qt(K, x)
        eigs.append(new_eig)
        eigvecs.append(x)
        shifts = shifts[(shifts > new_eig) * (shifts != shift)]

    print(find_unique(eigs)[::-1])
    fig, ax = plt.subplots()
    ax.plot(shifts)
    ax.axhline(lambda_)
    ax.axhline(-lambda_)
    fig.tight_layout()
    plt.show()


def main():
    a()
    b()
    c()
    d3()

if __name__ == "__main__":
    d4()
