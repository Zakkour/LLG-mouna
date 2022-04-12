import numpy as np
from param import NPTYPE


def interieure_Q(n, deltax):

    ct1 =  83   * deltax / ( 9. * 240 )
    ct2 = -23   * deltax / ( 9. * 240 )
    ct3 =  8.50 * deltax /        240
    ct4 =  5.75 * deltax /        240
    ct5 = -4.25 * deltax /        240

    ii = 3 * (n - 1)

    iid = [x for x in range(3, ii  , 1)]
    iim = [x for x in range(0, ii-3, 1)]
    iip = [x for x in range(6, ii+3, 1)]

    Q_1 = np.zeros((3*n, 3*n), dtype=NPTYPE)
    Q_1[iid, iid] = ct3 
    Q_1[iid, iim] = ct4
    Q_1[iid, iip] = ct5

    ia = [ii  , ii+1, ii+2]
    ib = [ii-3, ii-2, ii-1]
    Q_1[ia, ib] = ct1 
    Q_1[ia, ia] = ct2 

    return(Q_1)


def mat_Q(n, deltax, m_vec):

    Q_1 = interieure_Q(n, deltax)
    Q_2 = np.dot(Q_1 , m_vec)

    Q = np.zeros( (3*n, 3*n), dtype=NPTYPE )
    for ii in range(2, n+1, 1):

        iax = 3 * (ii - 1) 
        iay = iax + 1
        iaz = iax + 2

        qix = Q_2[iax]
        qiy = Q_2[iay]
        qiz = Q_2[iaz]

        Q[iax, iax - 2] = -qiz # -Q_2[iax+2]
        Q[iax, iax - 1] =  qiy #  Q_2[iax+1]
        Q[iay, iax - 3] =  qiz #  Q_2[iax+2]
        Q[iay, iax - 1] = -qix # -Q_2[iax]
        Q[iaz, iax - 3] = -qiy # -Q_2[iax+1]
        Q[iaz, iax - 2] =  qix #  Q_2[iax]
   
    return( Q )



#n=4
#deltax=1
#m_vec=np.random.random(3*n)
#Q_1=interieure_Q( deltax, n )
#Q=mat_Q(deltax,n,m_vec)
#print(Q)

