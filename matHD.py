import numpy as np
from param import NPTYPE


# ________________________________________________________________________________________________
#
#                                 build the 3n x 3n matrix HD
def build_HD(n):

    HD = -0.5 * np.eye(3*n, k=0, dtype=NPTYPE)

    for i in range(n):
        ii = 3*i
        HD[ii, ii] = 0.

    return(HD)
# ________________________________________________________________________________________________

