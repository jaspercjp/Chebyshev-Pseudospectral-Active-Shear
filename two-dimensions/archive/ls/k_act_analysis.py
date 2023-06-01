from shear2D import ev
import numpy as np
import matplotlib.pyplot as plt

_gammadot = 1
_tau = 1
# _tau_a = 0.1

# kl, kr are endpoint values of k
kl, kr = 0, 10
k_step = 0.1
ks = np.arange(kl,kr,k_step)
_tau_as = [100]
mev = [None] * len(ks)

for _tau_a in _tau_as:
    for i in range(len(ks)):
        k = ks[i]
        mev[i] = np.max(np.real(ev(k,_gammadot,_tau,_tau_a)))
    plt.plot(ks, mev, label="$\\tau_a$ = {}".format(_tau_a))
        
        #print(k, mev[i])
    
# fig, ax = plt.subplots()
# plt.set_title("Eigenvalues against $k$")
plt.xlabel("$k$")
plt.ylabel("$max(Re(\\sigma))$")
plt.title("Max real part of $\\sigma$ vs. $k$, $\\dot\\gamma={}$, $\\tau={}$".format(_gammadot,_tau))
plt.legend()
plt.show()