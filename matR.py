import numpy as np
from param import NPTYPE


def interieure_R(n, deltax):

    ii = 3 * (n - 1)

    ct1 = 2123 * deltax / ( 9 * 240 )
    ct2 = - 23 * deltax / ( 9 * 240 )
    ct3 =  203 * deltax / 240
    ct4 =  8.5 * deltax / 240

    R_1 = np.zeros( (3*n, 3*n), dtype=NPTYPE)

    ia = [0, 1, 2]
    ib = [3, 4, 5]
    R_1[ia, ia] = ct1
    R_1[ia, ib] = ct2 

    ia = [ii  , ii+1, ii+2]
    ib = [ii-3, ii-2, ii-1]
    R_1[ia, ib] = ct2
    R_1[ia, ia] = ct1

    iid = [x for x in range(3, ii  , 1)]
    iim = [x for x in range(0, ii-3, 1)]
    iip = [x for x in range(6, ii+3, 1)]
    R_1[iid, iid] = ct3
    R_1[iid, iim] = ct4
    R_1[iid, iip] = ct4

    return ( R_1 )


def mat_R(n, deltax, m_vec):

    R_1 = interieure_R(n, deltax)
    R_2 = np.dot( R_1 , m_vec)

    R = np.zeros( (3*n, 3*n), dtype=NPTYPE )

    for ii in range(1, n + 1, 1):

        iax = 3 * (ii - 1)
        iay = iax + 1
        iaz = iax + 2

        rix = R_2[iax]
        riy = R_2[iay]
        riz = R_2[iaz]

        R[iax, iay] = -riz
        R[iax, iaz] =  riy
        R[iay, iax] =  riz
        R[iay, iaz] = -rix
        R[iaz, iax] = -riy
        R[iaz, iay] =  rix

    return (R)

#n=4
#deltax=1
#m_vec=np.random.random(3*n)
#R_1=interieure_R( deltax, n )
#R=mat_R(deltax,n,m_vec)

#print(R)
