import numpy as np
from sklearn.preprocessing import normalize


def norm_2( v_vec , N ):
    #fonction qui calcule notre norme 2=( 1/3*N sum_1^N |x_i|^2)^0.5
    n = np.linalg.norm (v_vec)
    a = ( 3 * N) ** 0.5
    return ( n / a )



