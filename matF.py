import numpy as np
from param import NPTYPE

#------------------------------------------------------------------------------------------------
def interieure_F_1(n, deltax, F):

    dxF = F * deltax
    ct1 =  83   * dxF / (9. * 240)
    ct2 = -23   * dxF / (9. * 240)
    ct3 =  8.5  * dxF / 240  # diagonale
    ct4 =  5.75 * dxF / 240  # au dessous de diagonale
    ct5 = -4.25 * dxF / 240  # au dessus de diagonale

    ii = 3 * (n - 1)

    # indiquer les indices:
    iid = [x for x in range(3, ii, 1)]  # diagonale
    iim = [x for x in range(0, ii - 3, 1)]  # au dessous de diagonale
    iip = [x for x in range(6, ii + 3, 1)]  # au dessus de diagonale

    # remplir ces indices: avec cette methode on ne prend pas ni 1re ligne ni la dernière ligne
    F_11 = np.zeros((3 * n, 3 * n), dtype=NPTYPE)
    F_11[iid, iid] = ct3
    F_11[iid, iim] = ct4
    F_11[iid, iip] = ct5

    ia = [ii, ii + 1, ii + 2]
    ib = [ii - 3, ii - 2, ii - 1]
    F_11[ia, ib] = ct1
    F_11[ia, ia] = ct2

    return (F_11)


def mat_F_1(n, deltax, F, m_vec):

    F_11 = interieure_F_1(n, deltax, F)
    F_12 = np.dot(F_11, m_vec)

    F_1 = np.zeros((3 * n, 3 * n), dtype=NPTYPE)
    for ii in range(2, n + 1, 1):
        iax = 3 * (ii - 1)
        iay = iax + 1
        iaz = iax + 2

        qix = F_12[iax]
        qiy = F_12[iay]
        qiz = F_12[iaz]

        F_1[iax, iax - 2] = -qiz  # -Q_2[iax+2]
        F_1[iax, iax - 1] =  qiy  # Q_2[iax+1]
        F_1[iay, iax - 3] =  qiz  # Q_2[iax+2]
        F_1[iay, iax - 1] = -qix  # -Q_2[iax]
        F_1[iaz, iax - 3] = -qiy  # -Q_2[iax+1]
        F_1[iaz, iax - 2] =  qix  # Q_2[iax]

    return (F_1)

#on remarque que F1=F*Q
#------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------
def interieure_F_2(n, deltax, F):

    ii = 3 * (n - 1)

    dxF = F * deltax
    ct1 = 2123 * dxF / ( 9 * 240 )
    ct2 = - 23 * dxF / ( 9 * 240 )
    ct3 =  203 * dxF / 240
    ct4 =  8.5 * dxF / 240

    F_21 = np.zeros( (3*n, 3*n), dtype=NPTYPE)

    ia = [0, 1, 2]
    ib = [3, 4, 5]
    F_21[ia, ia] = ct1
    F_21[ia, ib] = ct2

    ia = [ii  , ii+1, ii+2]
    ib = [ii-3, ii-2, ii-1]
    F_21[ia, ib] = ct2
    F_21[ia, ia] = ct1

    iid = [x for x in range(3, ii  , 1)]
    iim = [x for x in range(0, ii-3, 1)]
    iip = [x for x in range(6, ii+3, 1)]
    F_21[iid, iid] = ct3
    F_21[iid, iim] = ct4
    F_21[iid, iip] = ct4

    return ( F_21 )


def mat_F_2(n, deltax, F, m_vec):

    F_21 = interieure_F_2(n, deltax, F)
    F_22 = np.dot( F_21 , m_vec)

    F_2 = np.zeros( (3*n, 3*n), dtype=NPTYPE )

    for ii in range(1, n + 1, 1):

        iax = 3 * (ii - 1)
        iay = iax + 1
        iaz = iax + 2

        rix = F_22[iax]
        riy = F_22[iay]
        riz = F_22[iaz]

        F_2[iax, iay] = -riz
        F_2[iax, iaz] =  riy
        F_2[iay, iax] =  riz
        F_2[iay, iaz] = -rix
        F_2[iaz, iax] = -riy
        F_2[iaz, iay] =  rix

    return (F_2)
#on remarque que F2=F*R
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------

def interieure_F_3(n, deltax, F):

    ii = 3 * (n - 1)

    dxF = F * deltax
    ct1 = -23   * dxF / ( 9. * 240 )
    ct2 =  83   * dxF / ( 9. * 240 )
    ct3 =  8.5  * dxF / 240
    ct4 =  5.75 * dxF / 240
    ct5 = -4.25 * dxF / 240

    F_31 = np.zeros( (3*n, 3*n), dtype=NPTYPE )

    ia = [0, 1, 2]
    ib = [3, 4, 5]
    F_31[ia, ia] = ct1
    F_31[ia, ib] = ct2

    iid = [x for x in range(3, ii  , 1)]
    iim = [x for x in range(0, ii-3, 1)]
    iip = [x for x in range(6, ii+3, 1)]
    F_31[iid, iid] = ct3
    F_31[iid, iip] = ct4
    F_31[iid, iim] = ct5

    return (F_31)


def mat_F_3(n, deltax, F, m_vec):

    F_31 = interieure_F_3(n, deltax, F)
    F_32 = np.dot( F_31 , m_vec )

    F_3 = np.zeros( (3*n, 3*n), dtype=NPTYPE )

    for ii in range(1, n, 1):

        iax = 3 * (ii - 1)
        iay = iax + 1
        iaz = iax + 2

        six = F_32[iax]
        siy = F_32[iay]
        siz = F_32[iaz]

        F_3[iax, iax + 4] = -siz
        F_3[iax, iax + 5] =  siy
        F_3[iay, iax + 3] =  siz
        F_3[iay, iax + 5] = -six
        F_3[iaz, iax + 3] = -siy
        F_3[iaz, iax + 4] =  six

    return ( F_3 )

#on remarque que F1=F*S
#------------------------------------------------------------------------------------------------




# n=4
# deltax=1
# m_vec=np.random.random(3*n)
# Q_1=interieure_Q( deltax, n )
# Q=mat_Q(deltax,n,m_vec)
# print(Q)

