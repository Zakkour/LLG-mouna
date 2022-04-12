
# 
#                   plot e(N)
# 

import matplotlib.pyplot as plt

# read N and e
ff = open("e_2", "r")
lines = ff.readlines()[1:]
N, e = [], []
for line in lines:
    ll = line.split()
    N.append(int  (ll[1]))
    e.append(float(ll[2]))


# start fig

fig = plt.figure(figsize=(8, 6)) # dimension 8 x 6

plt.xscale('log')  # plot in log scale
plt.yscale('log')  # plot in log scale

# x & y label 
plt.xlabel( 'N'
          , fontsize = 15
          , fontweight = 'bold'
          )
plt.ylabel('error'
          , fontsize = 15
          , fontweight = 'bold'
          )

# title, xtics, ytics, ...

# plot e(N)
#plt.plot( N[5:], e[5:]          # x, y
plt.plot( N[2:], e[2:]
        , color  = 'green'      # colors
        , marker = 'o'          
        , linestyle = 'dashed' 
        , linewidth = 2        
        , markersize = 12 )

# save
plt.savefig('e_n.pdf', dpi=300) #dpi pour qualité mieux


