#pour différent valeur de N et F calculer l'erreur et enregistrer les solutions approché dans les files m_i.

import scipy as sp
import numpy as np
import sys

from matHD import build_HD
from matLAMBDA import build_LAMBDA
from matM import matrice_M
from matQ import mat_Q
from matR import mat_R
from matS import mat_S
from matV import build_V
from matLAP import build_LAP
from solution_exacte import solt_exacte, Happ_tj
from norme_2 import norm_2

from sklearn.preprocessing import normalize

import time

from param import NPTYPE




def erreur(L, N, n, deltax, F, deltat, T_fin, save_m=None):
    t0 = time.time()

    print(' start calcul error for:')
    print('   L = {}'.format(L))
    print('   N = {}'.format(N))
    print('   F = {}'.format(F))
    sys.stdout.flush() #do8ri tbo3 ma tontor

    x = np.linspace(0, L, num=N, endpoint=True)
    m_vec0= np.array([ solt_exacte(0, i, L) for i in x ])
    m_vec1=np.array(m_vec0.flatten().tolist()) #vecteur complète pour m_0
    m_vec00=m_vec1[3:-3] #sans m^0_0 et m^0_N et comme vecteur

    M      = matrice_M(n, deltax)
    LAP    = build_LAP(n, deltax)
    HD     = build_HD(n)
    LAMBDA = build_LAMBDA(n)


    # ----------------------------------------
    # useful for VLAP matrix for LARGEN
    if LARGEN:
        dx2_inv = 1. / ( deltax * deltax )
        ct1 = -2. * dx2_inv / 3.
        ct2 =  2. * dx2_inv / 3.
        ct3 = -2. * dx2_inv
        ct4 =       dx2_inv
    
        mask1 = np.array([ 0, 1, 2 ])
        mask2 = np.array([ 3, 4, 5 ])
    
        ii_m = 3 * (n - 1)
        mask3 = np.array([ ii_m  , ii_m+1, ii_m+2 ])
        mask4 = np.array([ ii_m-3, ii_m-2, ii_m-1 ])
    
        mask5_d, mask5_m, mask5_p = [], [], []
        for i in range(3, ii_m, 3):
            mask5_d.extend([ (i  ,i), (i  , i+1), (i  , i+2)
                           , (i+1,i), (i+1, i+1), (i+1, i+2)
                           , (i+2,i), (i+2, i+1), (i+2, i+2) ])
            mask5_m.extend([ (i  ,i-3), (i  , i-2), (i  , i-1)
                           , (i+1,i-3), (i+1, i-2), (i+1, i-1)
                           , (i+2,i-3), (i+2, i-2), (i+2, i-1) ])
            mask5_p.extend([ (i  ,i+3), (i  , i+4), (i  , i+5)
                           , (i+1,i+3), (i+1, i+4), (i+1, i+5)
                           , (i+2,i+3), (i+2, i+4), (i+2, i+5) ])
    # ----------------------------------------

    pp = [3*i for i in range(n)] #[0, 3, 6, ...]

    m_vec_old = np.copy(m_vec00)

    t1 = 0.
    e1 = 0
    ii = 0
    while (t1 <= T_fin):
        t1 += deltat

        Q   = mat_Q(n, deltax, m_vec_old)
        R   = mat_R(n, deltax, m_vec_old)
        S   = mat_S(n, deltax, m_vec_old)
        QRS = Q + R + S

        QRSHD = -0.5 * QRS
        QRSHD[:, pp] = F * QRS[:, pp] #QRS.hd+F*QRS.LAMBDA=QRS.(hd+F*LAMBDA) yali huwe F 3a indice pp w -0.5
                                                                                        # bi be2i diagonale)

        V = build_V(n, deltax, np.dot(M, m_vec_old))

        if LARGEN:
            VLAP = np.zeros((3*n, 3*n), dtype=NPTYPE)
            VLAP[mask1[:, None], mask1] = ct1 * V[mask1[:, None], mask1]
            VLAP[mask1[:, None], mask2] = ct2 * V[mask1[:, None], mask1]
            VLAP[mask3[:, None], mask3] = ct1 * V[mask3[:, None], mask3]
            VLAP[mask3[:, None], mask4] = ct2 * V[mask3[:, None], mask3]
            VLAP[tuple(np.array(list(zip(*mask5_d))))] = ct3 * V[tuple(np.array(list(zip(*mask5_d))))]
            VLAP[tuple(np.array(list(zip(*mask5_m))))] = ct4 * V[tuple(np.array(list(zip(*mask5_d))))]
            VLAP[tuple(np.array(list(zip(*mask5_p))))] = ct4 * V[tuple(np.array(list(zip(*mask5_d))))]
        else:
            VLAP = V @ LAP

        MM = deltat * (VLAP + QRSHD)
        A  = M - QRS + MM
        B  = M - MM
        B1 = np.dot(B, m_vec_old) - 2 * deltat * Happ_tj(n, deltax, L, F, t1, m_vec_old)

        m_vec_new = np.linalg.solve(A, B1) # already parallelized  

        #renormalisation de m-vec-new
        for jj in range( 0, n ):
            y = m_vec_new[3 * jj : 3 * jj + 3]
            m_vec_new[3 * jj : 3 * jj + 3] = normalize(y[:, np.newaxis], axis=0).ravel()

        m_vec_old = np.copy(m_vec_new)  # sans m^j_0 et m^j_N-1 c'est à dire 1re et dernier élément

        mj_0 = 4 * m_vec_old[:3] / 3 - m_vec_old[3:6] / 3  # 1ere élément
        mj_0 = normalize(mj_0[:, np.newaxis], axis=0).ravel()
        mj_N = 4 * m_vec_old[-3:] /3 - m_vec_old[-6:-3] / 3  # dernier élément
        mj_N = normalize(mj_N[:, np.newaxis], axis=0).ravel()
        mj  = np.insert(m_vec_old, 0, mj_0)  # ajouter le 1er     élément
        m_j = np.append(mj, mj_N)            # ajouter le dernier élément

        m_ex  = np.array([ solt_exacte(t1, i, L) for i in x])
        m_ex1 = np.array(m_ex.flatten().tolist())
        e1    = max(e1, norm_2(m_ex1-m_j, N))

        ii += 1

    print(' nb of iter = {}, cpu tot time = {} min'.format(ii, (time.time()-t0)/60.))
    print(' error after {} sec = {}\n'.format(t1, e1)) #t1: T_fin (wa2t phisique)
    sys.stdout.flush()

    if save_m:
        print(' saving solution in {}'.format(save_m))
        with open(save_m, 'w') as ff:
            for i in range(3*N):
                ff.write('{:+e}\n'.format(m_j[i]))

    return e1


T_fin = 1.
L = 4

N_list = [10,20,30,50,100,200,400,800]
F_list = [ -2 ]

file_i = 0
for N in N_list:

    n = N-2
    deltax = L / (n+1)
    deltat = deltax / 10
    LARGEN = ( True if N >= 1000
               else False )

    for F in F_list:
        file_i += 1
        file_m = 'res/m_' + str(file_i) #à la place de file_m je met save_m
        # _ ==> do not need the output of erreur (ye3ni fct bteshte8el bs ma betrod shay)
        _ = erreur(L, N, n, deltax, F, deltat, T_fin, save_m=file_m)




