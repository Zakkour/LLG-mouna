import numpy as np
from param import NPTYPE


def matrice_M(n, deltax):

    M = ( 22. * np.eye(3*n, k = 0, dtype = NPTYPE)
              + np.eye(3*n, k = 3, dtype = NPTYPE)
              + np.eye(3*n, k =-3, dtype = NPTYPE) )

    M[0, 0] = 70 / 3.
    M[1, 1] = 70 / 3.
    M[2, 2] = 70 / 3.

    M[0, 3] = 2. / 3.
    M[1, 4] = 2. / 3.
    M[2, 5] = 2. / 3.

    ii = 3 * ( n - 1 )

    M[ii+0, ii-3] = 2. / 3.
    M[ii+1, ii-2] = 2. / 3.
    M[ii+2, ii-1] = 2. / 3.

    M[ii+0, ii+0] = 70 / 3.
    M[ii+1, ii+1] = 70 / 3.
    M[ii+2, ii+2] = 70 / 3.

    M = deltax * M / 24

    return( M )

#deltax = 1
#n = 4
#M=matrice_M(deltax,n)
#print(M)








