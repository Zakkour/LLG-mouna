#trace la solution approché et comment varie aucours du temps.

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



from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt


def plt_err(L, N, n, deltax, F, deltat, T_fin):

    x = np.linspace(0, L, num=N, endpoint=True)
    m_vec0= np.array([ solt_exacte(0, i, L) for i in x ])
    m_vec1=np.array(m_vec0.flatten().tolist()) 
    m_vec00=m_vec1[3:-3]

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
    
        ii_m = 3 * (n - 1)

        mask1 = np.array([ 0, 1, 2 ])
        mask2 = np.array([ 3, 4, 5 ])
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
 

    # plot
    plt.ion()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x axis'); ax.set_ylabel('y axis'); ax.set_zlabel('z axis')
    ax.set_xlim([0., L+0.1]); ax.set_ylim([0., L+0.1]); ax.set_zlim([0., L+0.1])
    soa = []
    for i in range(0, N, 10):
        soa.append([ i*deltax, 0, 0, m_vec1[3*i], m_vec1[3*i+1], m_vec1[3*i+2] ])
    x, y, z, u, v, w = zip(*soa)
    ax.quiver(x, y, z, u, v, w)
    ax.view_init(20, -90) # elevated angle and horizontal angle (in degrees)
    plt.show()


    t1 = 0.
    while (t1 <= T_fin):
        t1 += deltat

        Q = mat_Q(n, deltax, m_vec_old)
        R = mat_R(n, deltax, m_vec_old)
        S = mat_S(n, deltax, m_vec_old)
        V = build_V(n, deltax, np.dot(M, m_vec_old))

        QRS = Q + R + S

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

        QRSHD = -0.5 * QRS
        QRSHD[:, pp] = 0.

        FL = F * np.dot(QRS, LAMBDA)

        MM = deltat * (VLAP + QRSHD + FL)
        A  = M - QRS + MM
        B  = M - MM
        B1 = np.dot(B, m_vec_old) - 2 * deltat * Happ_tj(n, deltax, L, F, t1, m_vec_old)

        m_vec_new = np.linalg.solve( A , B1 )

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

        soa = [] #pour indiquer le paramètre dans le dessin
        for i in range(0, N, 10):
            soa.append([ i*deltax, 0, 0, m_j[3*i], m_j[3*i+1], m_j[3*i+2] ])
        x, y, z, u, v, w = zip(*soa)
        ax.quiver(x, y, z, u, v, w)
        ax.set_xlabel('x axis'); ax.set_ylabel('y axis'); ax.set_zlabel('z axis')
        ax.set_xlim([0., L+0.1]); ax.set_ylim([0., L+0.1]); ax.set_zlim([0., L+0.1])
        ax.view_init(20, -90) # elevated angle and horizontal angle (in degrees)
        plt.draw()
        plt.pause(0.001)
        plt.cla() #fig.clear btemhi kolshi ma3 axes




T_fin = 1.
L = 4

N = 100
F = 10

n = N-2
deltax = L / (n+1)
deltat = deltax / 10
LARGEN = ( True if N >= 1000
           else False )

plt_err(L, N, n, deltax, F, deltat, T_fin)


