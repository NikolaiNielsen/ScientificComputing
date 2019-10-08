import numpy as np
import matplotlib.pyplot as plt
from f4 import *


def test_rk4():

    # Parameters:
    # a1, a2, b1, b2, b3, c1, c2, d1, e, r1, r2, r3, r4, p1, p2, q, r
    params = [10, 5, 5, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 5, 100, 100]
    params = np.array(params)
    x0 = [0.01, 0, 0, 0]
    x0 = np.array(x0)
    x, t = sim(f, x0, params, dt=5e-4, N=350)
    fig, ax = plt.subplots()
    x = x.squeeze().T
    ax.plot(t, x)
    plt.show()


def main():
    test_rk4()


if __name__ == "__main__":
    main()
