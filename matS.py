import numpy as np
from param import NPTYPE


def interieure_S(n, deltax):

    ii = 3 * (n - 1)

    ct1 = -23 * deltax / ( 9. * 240 )
    ct2 =  83 * deltax / ( 9. * 240 )

    ct3 =  8.5  * deltax / 240
    ct4 =  5.75 * deltax / 240
    ct5 = -4.25 * deltax / 240

    S_1 = np.zeros( (3*n, 3*n), dtype=NPTYPE )

    ia = [0, 1, 2]
    ib = [3, 4, 5]
    S_1[ia, ia] = ct1 
    S_1[ia, ib] = ct2 

    iid = [x for x in range(3, ii  , 1)]
    iim = [x for x in range(0, ii-3, 1)]
    iip = [x for x in range(6, ii+3, 1)]
    S_1[iid, iid] = ct3
    S_1[iid, iip] = ct4
    S_1[iid, iim] = ct5

    return (S_1)


def mat_S(n, deltax, m_vec):

    S_1 = interieure_S(n, deltax)
    S_2 = np.dot( S_1 , m_vec )

    S = np.zeros( (3*n, 3*n), dtype=NPTYPE )

    for ii in range(1, n, 1):

        iax = 3 * (ii - 1)
        iay = iax + 1
        iaz = iax + 2

        six = S_2[iax]
        siy = S_2[iay]
        siz = S_2[iaz]

        S[iax, iax + 4] = -siz
        S[iax, iax + 5] =  siy
        S[iay, iax + 3] =  siz
        S[iay, iax + 5] = -six
        S[iaz, iax + 3] = -siy
        S[iaz, iax + 4] =  six

    return ( S )

#n=4
#deltax=1
#m_vec=np.ones(3*n)
#m_vec=np.random.random(3*n)
#S_1=interieure_S( deltax, n )
#S=mat_S(deltax,n,m_vec)

#print(S_1)

