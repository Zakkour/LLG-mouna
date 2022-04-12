
# 
#                   plot m_j
# 

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np

L = 4.
N = 10 
n = N-2
deltax = L / (n+1)

# ---------------------------
# read results

ff = open("m_1", "r")
lines = ff.readlines()
m1 = []
for line in lines:
    m1.append(float(line))

ff = open("m_2", "r")
lines = ff.readlines()
m2 = []
for line in lines:
    m2.append(float(line))

ff = open("m_3", "r")
lines = ff.readlines()
m3 = []
for line in lines:
    m3.append(float(line))

# ---------------------------


fig = plt.figure()
ax = fig.gca(projection='3d')

M1, M2, M3 = [], [], []
for i in range(0, N, 1):
    M1.append([ i*deltax, 0, 0, m1[3*i], m1[3*i+1], m1[3*i+2] ])
    M2.append([ i*deltax, 0, 0, m2[3*i], m2[3*i+1], m2[3*i+2] ])
    M3.append([ i*deltax, 0, 0, m3[3*i], m3[3*i+1], m3[3*i+2] ])
x, y, z, u1, v1, w1 = zip(*M1)
_, _, _, u2, v2, w2 = zip(*M2) #x,y,z déja 5edton bl sater yali fa2
_, _, _, u3, v3, w3 = zip(*M3)

ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')

ax.set_ylim([0., L])
ax.set_zlim([0., L])

ax.quiver( x, y, z, u1, v1, w1, color='r', label='F=-2') #lezem na2i F sah
ax.quiver( x, y, z, u2, v2, w2, color='b', label='F=0')
ax.quiver( x, y, z, u3, v3, w3, color='g', label='F=+2')

ax.legend()

# elevated angle and horizontal angle (in degrees)
ax.view_init(20, -80)
plt.savefig('vec.pdf')

plt.show()





