import numpy as np
from param import NPTYPE

# ________________________________________________________________________________________________
#
#                                 build the 3n x 3n matrix LAP
def build_LAP(n, deltax):

    LAP = np.zeros( (3*n, 3*n), dtype=NPTYPE )

    ii = 3 * (n - 1)
    dx2_inv = 1. / ( deltax * deltax )

    ct1 =  2. * dx2_inv / 3.
    ct2 = -2. * dx2_inv / 3. 
    ct3 =       dx2_inv 
    ct4 = -2. * dx2_inv

    ia  = [0, 1, 2]
    ib  = [3, 4, 5]
    LAP[ia, ia] = ct2
    LAP[ia, ib] = ct1

    ia = [ii  , ii+1, ii+2]
    ib = [ii-3, ii-2, ii-1]
    LAP[ia, ia] = ct2
    LAP[ia, ib] = ct1

    iid = [x for x in range(3, ii  , 1)]
    iim = [x for x in range(0, ii-3, 1)]
    iip = [x for x in range(6, ii+3, 1)]
    LAP[iid, iim] = ct3
    LAP[iid, iip] = ct3
    LAP[iid, iid] = ct4 #ct2

    return (LAP)
# ________________________________________________________________________________________________


#LAP=build_LAP()
#print(LAP)
