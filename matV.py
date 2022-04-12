import numpy as np
from param import NPTYPE



# ________________________________________________________________________________________________
#
#                                 build the 3n x 3n matrix V
def build_V(n, deltax, p_vec):

    V = np.zeros( (3*n, 3*n), dtype=NPTYPE )
    for ii in range(1, n+1, 1):

        iax = 3 * ( ii - 1 )
        iay = iax + 1
        iaz = iax + 2

        vix = p_vec[iax]
        viy = p_vec[iay]
        viz = p_vec[iaz]

        V[iax, iay] = -viz
        V[iax, iaz] =  viy
        V[iay, iax] =  viz
        V[iay, iaz] = -vix
        V[iaz, iax] = -viy
        V[iaz, iay] =  vix

    return(V)
# ________________________________________________________________________________________________

