import numpy as np
from functions import *

A = np.array([[4, 0.5, 0], [0.6, 5, 0.6], [0, 0.5, 3]])
centers, radii = gershgorin_disks(A)

max_eig = power_iteration(A)
max_eig2 = power_iteration(A, rayleigh=True)
print(max_eig)
print(max_eig2)

max_eig = rayleigh_quotient_iteration(A)
print(max_eig)
# print(np.linalg.eigvals(A)[0])