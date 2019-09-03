import numpy as np
from watermatrices import Amat, Bmat, yvec
from functions import *

E = np.vstack((np.hstack((Amat, Bmat)), np.hstack((Bmat, Amat))))
diag = np.array((*[1]*7, *[-1]*7))
S = np.diag(diag)
z = np.array([*yvec, *-yvec])

omega = np.array((1.300,  1.607, 3.000))
# print("Answer a2:")
# for o in omega:
#     M = E - o*S
#     print(f"Condition number for omega={o}")
#     print(cond(M))

A = np.array([[1, 2, 2], [4, 4, 2], [4, 6, 4]])
L, U = lu_factorize(A)