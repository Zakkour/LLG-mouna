import math

import numpy as np
from usefull import alpha_i, beta_i, gama_i, line_1, line_2 ,line_3, line_4
from scipy import integrate
#from scipy.integrate import quad_vec
from scipy.integrate import fixed_quad


def solt_exacte(t, x, L):
    # definir une solution exacte de norme = 1
    teta = 0.0625 * x * x * (x-L) * (x-L)
    y    = np.array( [ math.sin(t) * math.cos(teta) , math.sin(t) * math.sin(teta) , math.cos(t) ] )
    return (y)


#derivé de solt_exacte par rapport au temps
def dt_solt_exacte(t, x, L):
    teta = 0.0625 * x * x * (x-L) * (x-L)
    y = np.array([math.cos(t) * math.cos(teta), math.cos(t) * math.sin(teta), - math.sin(t)])
    return (y)


#dérivée de solt_exacte 2 fois par rapport à x
def dxx_solt_exacte(t, x, L):

    teta     = 0.0625 * x * x * (x-L) * (x-L)
    dx_teta  = 0.125 * x * (x-L) * (2*x-L)
    dxx_teta = 0.125 * (6*x*x - 6*x*L + L*L)

    y = np.array([ math.sin(t) * (-dxx_teta *  math.sin(teta) - dx_teta * dx_teta * math.cos(teta))
                 , math.sin(t) * ( dxx_teta *  math.cos(teta) - dx_teta * dx_teta * math.sin(teta))
                 , 0 ])
    return y


# ----------------------------------------------------
# costruction du champs appliquée

def Happ(x, t, L, F):

    cost = math.cos(t) 
    sint = math.sin(t) 

    teta     = 0.0625 * x * x * (x-L) * (x-L)
    dx_teta  = 0.125 * x * (x-L) * (2*x-L)
    dxx_teta = 0.125 * (6*x*x - 6*x*L + L*L)
    cosO     = math.cos(teta) 
    sinO     = math.sin(teta) 

    m = np.array([ sint * cosO 
                 , sint * sinO
                 , cost ])

    hd = np.array([ F*m[0], -0.5*m[1] , -0.5*m[2] ])

    dt_m = np.array([ cost * cosO 
                    , cost * sinO
                    , -sint ])

    dxx_m = np.array([ sint * (-dxx_teta * sinO - dx_teta * dx_teta * cosO )
                     , sint * ( dxx_teta * cosO - dx_teta * dx_teta * sinO )
                     , 0 ])

    ha = 0.5 * dt_m + np.cross( m, 0.5*dt_m + np.cross(m, hd + dxx_m) )

    return ha
# ----------------------------------------------------



# ----------------------------------------------------

def dhi_11(x, tj, xi, xim, xip, deltax, L,F):
    ha = Happ(x, tj, L, F)[0]
    aa = alpha_i(x, xi, xim, xip, deltax)
    return( aa * ha )
def dhi_12(x, tj, xi, xim, xip, deltax, L,F):
    ha = Happ(x, tj, L, F)[1]
    aa = alpha_i(x, xi, xim, xip, deltax)
    return( aa * ha )
def dhi_13(x,  tj , xi, xim, xip, deltax, L,F):
    ha = Happ(x, tj, L, F)[2]
    aa = alpha_i(x, xi, xim, xip, deltax)
    return( aa * ha )

def dhi_1(x, tj, xi, xim, xip, deltax, L,F):
    ha = Happ(x, tj, L, F)
    aa = alpha_i(x, xi, xim, xip, deltax)
    return( aa * ha )


def dhi_1_vec(x, tj, xi, xim, xip, deltax, L, F):

    cost = math.cos(tj) 
    sint = math.sin(tj) 

    teta     = 0.0625 * x * x * (x-L) * (x-L)
    dx_teta  = 0.125 * x * (x-L) * (2*x-L)
    dxx_teta = 0.125 * (6*x*x - 6*x*L + L*L)
    cosO     = np.cos(teta) 
    sinO     = np.sin(teta) 

    m     = np.zeros( (x.shape[0], 3) )
    hd    = np.zeros( (x.shape[0], 3) )
    dt_m  = np.zeros( (x.shape[0], 3) )
    dxx_m = np.zeros( (x.shape[0], 3) )

    m[:, 0] = sint * cosO[:]
    m[:, 1] = sint * sinO[:]
    m[:, 2] = cost

    hd[:, 0] =  F   * sint * cosO[:]
    hd[:, 1] = -0.5 * sint * sinO[:]
    hd[:, 2] = -0.5 * cost       

    dt_m[:, 0] = cost * cosO[:]
    dt_m[:, 1] = cost * sinO[:]
    dt_m[:, 2] = -sint

    dxx_m[:, 0] = sint * (-dxx_teta[:] * sinO[:] - dx_teta[:] * dx_teta[:] * cosO[:])
    dxx_m[:, 1] = sint * ( dxx_teta[:] * cosO[:] - dx_teta[:] * dx_teta[:] * sinO[:])
    dxx_m[:, 2] = 0.

    ha = np.zeros((x.shape[0], 3))
    ha = ( 0.5 * dt_m + np.cross( m, 
           0.5 * dt_m + np.cross( m, hd + dxx_m) ) )

    aa = 0.5 * ( x - xi ) * ( x - xip ) / (deltax*deltax)

    res = ha * aa[:, np.newaxis]
    return res.T

# ----------------------------------------------------




# ----------------------------------------------------

def dhi_vec(x, tj, xi, xim, xip, deltax, L, F):

    cost = math.cos(tj) 
    sint = math.sin(tj) 

    teta     = 0.0625 * x * x * (x-L) * (x-L)
    dx_teta  = 0.125 * x * (x-L) * (2*x-L)
    dxx_teta = 0.125 * (6*x*x - 6*x*L + L*L)
    cosO     = np.cos(teta) 
    sinO     = np.sin(teta) 

    m     = np.zeros( (x.shape[0], 3) )
    hd    = np.zeros( (x.shape[0], 3) )
    dt_m  = np.zeros( (x.shape[0], 3) )
    dxx_m = np.zeros( (x.shape[0], 3) )

    m[:, 0] = sint * cosO[:]
    m[:, 1] = sint * sinO[:]
    m[:, 2] = cost

    hd[:, 0] =  F   * sint * cosO[:]
    hd[:, 1] = -0.5 * sint * sinO[:]
    hd[:, 2] = -0.5 * cost       

    dt_m[:, 0] = cost * cosO[:]
    dt_m[:, 1] = cost * sinO[:]
    dt_m[:, 2] = -sint

    dxx_m[:, 0] = sint * (-dxx_teta[:] * sinO[:] - dx_teta[:] * dx_teta[:] * cosO[:])
    dxx_m[:, 1] = sint * ( dxx_teta[:] * cosO[:] - dx_teta[:] * dx_teta[:] * sinO[:])
    dxx_m[:, 2] = 0.

    ha = np.zeros((x.shape[0], 3))
    ha = ( 0.5 * dt_m + np.cross( m, 
           0.5 * dt_m + np.cross( m, hd + dxx_m) ) )

    tmp = 1. / (deltax*deltax)

    aa =  0.5 * tmp * ( x - xi )  * ( x - xip ) 
    bb = - tmp * ( x - xim ) * ( x - xip ) 
    cc =  0.5 * tmp * ( x - xi )  * ( x - xim )

    resa = ha * aa[:, np.newaxis]
    resb = ha * bb[:, np.newaxis]
    resc = ha * cc[:, np.newaxis]

    res = np.concatenate( (resa.T, resb.T, resc.T), axis=0)

    return res
# ----------------------------------------------------




# ----------------------------------------------------

def dhi_21(x,  tj , xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[0]
    bb = beta_i(x, xi, xim, xip, deltax)
    return bb * ha
def dhi_22(x,  tj , xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[1]
    bb = beta_i(x, xi, xim, xip, deltax)
    return bb * ha
def dhi_23(x,  tj , xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[2]
    bb = beta_i(x, xi, xim, xip, deltax)
    y  = bb * ha
    return (y)

def dhi_2(x, tj, xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)
    bb = beta_i(x, xi, xim, xip, deltax)
    return bb * ha

# ----------------------------------------------------



# ----------------------------------------------------

def dhi_31(x,  tj , xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[0]
    cc = gama_i(x, xi, xim, xip, deltax)
    return cc * ha
def dhi_32(x, tj, xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[1]
    cc = gama_i(x, xi, xim, xip, deltax)
    return cc * ha
def dhi_33(x, tj , xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[2]
    cc = gama_i(x, xi, xim, xip, deltax)
    return cc * ha

def dhi_3(x, tj, xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)
    cc = gama_i(x, xi, xim, xip, deltax)
    return cc * ha

# ----------------------------------------------------


def dh_41(x, tj, xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[0]
    l1 = line_1(x, xi, xim, xip, deltax)
    return l1 * ha
def dh_42(x, tj, xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[1]
    l1 = line_1(x, xi, xim, xip, deltax)
    return l1 * ha
def dh_43(x, tj , xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[2]
    l1 = line_1(x, xi, xim, xip, deltax)
    return l1 * ha

def dh_51(x, tj, xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[0]
    l2 = line_2(x, xi, xim, xip, deltax)
    return l2 * ha
def dh_52(x, tj , xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[1]
    l2 = line_2(x, xi, xim, xip, deltax)
    return l2 * ha
def dh_53(x, tj , xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[2]
    l2 = line_2(x, xi, xim, xip, deltax)
    return l2 * ha

def dh_61(x, tj, xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[0]
    l3 = line_3(x, xi, xim, xip, deltax)
    return l3 * ha
def dh_62(x, tj, xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[1]
    l3 = line_3(x, xi, xim, xip, deltax)
    return l3 * ha
def dh_63(x, tj , xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[2]
    l3 = line_3(x, xi, xim, xip, deltax)
    return l3 * ha

def dh_71(x, tj, xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[0]
    l4 = line_4(x, xi, xim, xip, deltax)
    return l4 * ha
def dh_72(x, tj, xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[1]
    l4 = line_4(x, xi, xim, xip, deltax)
    return l4 * ha
def dh_73(x, tj , xi, xim, xip, deltax, L, F):
    ha = Happ(x, tj, L, F)[2]
    l4 = line_4(x, xi, xim, xip, deltax)
    return l4 * ha






def Happ_tj(n, deltax, L, F, tj, m_vec):
    #cette fonction donne une vecteur contient \int_wi u^j_i\v Happ

    ha_j = np.zeros( 3 * n )

    # -------------------------------------------------------------------------------
    #1re élement
    ii = 1

    xi  = ii * deltax
    xim = xi - deltax
    xip = xi + deltax

    # bounds
    wim = xi - 0.5 * deltax
    wip = xi + 0.5 * deltax

    integ_41, _ = integrate.quad(dh_41, wim, wip, args=(tj, xi, xim, xip, deltax, L, F))
    integ_42, _ = integrate.quad(dh_42, wim, wip, args=(tj, xi, xim, xip, deltax, L, F))
    integ_43, _ = integrate.quad(dh_43, wim, wip, args=(tj, xi, xim, xip, deltax, L, F))
    integ_51, _ = integrate.quad(dh_51, wim, wip, args=(tj, xi, xim, xip, deltax, L, F))
    integ_52, _ = integrate.quad(dh_52, wim, wip, args=(tj, xi, xim, xip, deltax, L, F))
    integ_53, _ = integrate.quad(dh_53, wim, wip, args=(tj, xi, xim, xip, deltax, L, F))

    int_4 = np.array( [integ_41, integ_42, integ_43 ] )
    int_5 = np.array( [integ_51, integ_52, integ_53] )

    m1 = m_vec[0:3]
    m2 = m_vec[3:6]
    ha_j[0:3] = np.cross( m1 , int_4 ) + np.cross( m2 , int_5 )

    # -------------------------------------------------------------------------------
    # au milieu

    deltax_div2 = 0.5 * deltax

    for ii in range(2, n, 1):

        xi  = ii * deltax
        xim = xi - deltax
        xip = xi + deltax

        # bounds
        wim = xi - deltax_div2
        wip = xi + deltax_div2 

        int_t, _  = fixed_quad(dhi_vec, wim, wip, args=(tj, xi, xim, xip, deltax, L, F), n=10)

        ia  = 3 * (ii - 1)
        mi  = m_vec[ia:ia+3]
        mim = m_vec[ia-3:ia]
        mip = m_vec[ia+3:ia+6]

        ha_j[ia:ia+3] = ( np.cross(mim, int_t[:3]) 
                        + np.cross(mi , int_t[3:6]) 
                        + np.cross(mip, int_t[6:]) )

    # -------------------------------------------------------------------------------
    #last term
    ii  = n
    xi  = ii * deltax
    xim = xi - deltax
    xip = xi + deltax

    # bounds
    wim = xi - 0.5 * deltax
    wip = xi + 0.5 * deltax

    integ_61, _ = integrate.quad(dh_61, wim, wip, args=(tj, xi, xim, xip, deltax, L, F))
    integ_62, _ = integrate.quad(dh_62, wim, wip, args=(tj, xi, xim, xip, deltax, L, F))
    integ_63, _ = integrate.quad(dh_63, wim, wip, args=(tj, xi, xim, xip, deltax, L, F))
    integ_71, _ = integrate.quad(dh_71, wim, wip, args=(tj, xi, xim, xip, deltax, L, F))
    integ_72, _ = integrate.quad(dh_72, wim, wip, args=(tj, xi, xim, xip, deltax, L, F))
    integ_73, _ = integrate.quad(dh_73, wim, wip, args=(tj, xi, xim, xip, deltax, L, F))

    int_6 = np.array([integ_61, integ_62, integ_63])
    int_7 = np.array([integ_71, integ_72, integ_73])

    ia = 3 * (ii - 1) # ia= 3*(n-1)

    m3 = m_vec[ia:ia + 3] # [3*(n-1) : 3*n] = m_n
    m4 = m_vec[ia - 3:ia] # [3*n-6 : 3*n-3] = m_(n-1)

    ha_j[ia:ia + 3] = np.cross(m3, int_7) + np.cross(m4, int_6)

    return (ha_j)













