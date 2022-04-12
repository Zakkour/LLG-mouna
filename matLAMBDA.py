import numpy as np
from param import NPTYPE


# ________________________________________________________________________________________________
#
#                                 build the 3n x 3n matrix LAMBDA
def build_LAMBDA(n):

    LAMBDA = np.zeros( (3*n, 3*n), dtype=NPTYPE)

    ii = [3*i for i in range(n)]
    LAMBDA[ii, ii] = 1.

    return(LAMBDA)
# ________________________________________________________________________________________________


