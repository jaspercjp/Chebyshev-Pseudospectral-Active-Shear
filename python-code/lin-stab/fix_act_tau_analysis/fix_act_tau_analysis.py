from shear2D import ev
import numpy as np
import matplotlib.pyplot as plt

_tau = 1
# _tau_a = 0.195 try this value later
_tau_a = 5

# kl, kr are endpoint values of k
# kl, kr, k_step = 0, 10, 0.1
kl, kr, k_step = 0, 0.3,0.1 
ks = np.arange(kl,kr,k_step)

# gdl, gdr, gd_step = 0.01,3.01,0.1
gdl, gdr, gd_step = 0.01,0.3,0.1
gds = np.arange(gdl,gdr,gd_step)

# array to store max real parts of eigenvalues
mev = [[None] * len(ks)] * len(gds)

print("Doing linear stability analysis for fixed tau={} and tau_a={}".format(_tau,_tau_a))
for i in range(len(gds)):
    gd = gds[i]
    for j in range(len(ks)):
        k = ks[j]
        mev[i][j] = np.max(np.real(ev(k,gd,_tau,_tau_a)))
    print("{}/{} completed ...".format(i+1, len(gds)))
print(np.array(mev))
        #print(k, mev[i])
        
# fig, ax = plt.subplots()
# plt.set_title("Eigenvalues against $k$")
# plt.xlabel("$k$")
# plt.ylabel("$max(Re(\\sigma))$")
# plt.title("Max real part of $\\sigma$ vs. $k$, $\\dot\\gamma={}$, $\\tau={}$".format(_gammadot,_tau))
# plt.legend()
# plt.show()