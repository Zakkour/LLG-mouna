import numpy as np
from scipy import integrate

# func to calculate alpha_i
def alpha_i(x, xi, xim, xip, deltax):
    y = 0.5 * ( x - xi ) * ( x - xip ) / deltax**2
    return(y)

# func to calculate beta_i
def beta_i(x, xi, xim, xip, deltax):
    y = - ( x - xim ) * ( x - xip ) / deltax**2
    return(y)

# func to calculate gama_i
def gama_i(x, xi, xim, xip, deltax):
    y = 0.5 * ( x - xi ) * ( x - xim ) / deltax**2
    return(y)

# func to calculate alpha_i**2
def alphaa_i(x, xi, xim, xip, deltax):
    aa = alpha_i(x, xi, xim, xip)
    return(aa*aa)

# func to calculate beta_i**2
def betaa_i(x, xi, xim, xip, deltax):
    bb = beta_i(x, xi, xim, xip)
    return(bb*bb)

# func to calculate gama_i**2
def gamaa_i(x, xi, xim, xip, deltax):
    cc = gama_i(x, xi,  xim, xip)
    return(cc*cc)

# func to calculate alpha_i*beta_i
def alphbeta_i(x, xi, xim, xip, deltax):
    aa = alpha_i(x, xi, xim, xip)
    bb = beta_i(x, xi, xim, xip)
    return(aa*bb)

# func to calculate alpha_i*gama_i
def alphgama_i(x, xi, xim, xip, deltax):
    aa = alpha_i(x, xi, xim, xip)
    cc = gama_i(x, xi, xim, xip)
    return(aa*cc)

# func to calculate beta_i*gama_i
def betgama_i(x, xi, xim, xip, deltax):
    bb = beta_i(x, xi, xim, xip, deltax)
    cc = gama_i(x, xi, xim, xip, deltax)
    return(bb*cc)

#4/3*alpha_i+beta_i
def line_1(x, xi, xim, xip, deltax):
    aa = alpha_i(x, xi, xim, xip, deltax)
    bb = beta_i(x, xi, xim, xip, deltax)
    return(4*aa/3+bb)

# -1_3*alpha_i+gama_i
def line_2(x, xi, xim, xip, deltax):
    aa = alpha_i(x, xi, xim, xip, deltax)
    cc = gama_i(x, xi, xim, xip, deltax)
    return(-aa/3+cc)

#alpha_i-1/3*gama_i
def line_3(x, xi, xim, xip, deltax):
    aa = alpha_i(x, xi, xim, xip, deltax)
    cc = gama_i(x, xi, xim, xip, deltax)
    return(aa-cc/3)

# beta_i+4/3*gama_i
def line_4(x, xi, xim, xip, deltax):
    cc = gama_i(x, xi, xim, xip, deltax)
    bb = beta_i(x, xi, xim, xip, deltax)
    return(bb+4*cc/3)

#(4/3*alpha_i+beta_i)**2
def line_11(x, xi, xim, xip, deltax):
    l1=line_1(x, xi, xim, xip, deltax)
    return(l1*l1)

#(4/3*alpha_i+beta_i)*(-1_3*alpha_i+gama_i)
def line_12(x, xi, xim, xip, deltax):
    l1 = line_1(x, xi, xim, xip, deltax)
    l2 = line_2(x, xi, xim, xip, deltax)
    return (l1*l2)

#(-1_3*alpha_i+gama_i)**2
def line_22 (x,  xi,xim, xip, deltax):
    l2=line_2(x, xi, xim, xip, deltax)
    return (l2*l2)


#(alpha_i-1/3*gama)**2
def line_33(x, xi, xim, xip, deltax):
    l3=line_3(x, xi, xim, xip, deltax)
    return(l3*l3)

#(alph  -1/3*gama)*(beta-i+4/3*gama_i)
def line_34(x, xi, xim, xip, deltax):
    l3 = line_3(x, xi, xim, xip, deltax)
    l4=line_4(x,  xi,xim, xip, deltax)
    return(l3*l4)
#(beta-i+4/3*gama_i)**2
def line_44(x, xi, xim, xip, deltax):
    l4=line_4(x, xi, xim, xip, deltax)
    return (l4*l4)


