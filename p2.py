import numpy as np
from f2 import *

A = np.array([[4, 0.5, 0], [0.6, 5, 0.6], [0, 0.5, 3]])
centers, radii = gershgorin(A)

max_eig = power_iterate(A)
max_eig2 = power_iterate(A, rayleigh=True)
print(max_eig)
print(max_eig2)

max_eig = rayleigh_iterate(A)
print(max_eig)
# print(np.linalg.eigvals(A)[0])