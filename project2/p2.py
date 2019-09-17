import numpy as np
from functions import *

A = np.array([[4, 0.5, 0], [0.6, 5, 0.6], [0, 0.5, 3]])
centers, radii = gershgorin_disks(A)
