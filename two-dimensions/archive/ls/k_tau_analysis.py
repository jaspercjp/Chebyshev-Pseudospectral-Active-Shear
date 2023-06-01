from shear2D import ev
import numpy as np
import matplotlib.pyplot as plt

_tau_a = 10
_gammadot = 1.0

taus = [1,10,50,100]
kl, kr = 0, 10
k_step = 0.1
ks = np.arange(kl,kr,k_step)
mev = [None] * len(ks)

for _tau in taus:
    for i in range(len(ks)):
        k = ks[i]
        mev[i] = np.max(np.real(ev(k, _gammadot, _tau, _tau_a)))
    plt.plot(ks, mev, label="$\\tau$ = {}".format(_tau))

plt.xlabel("$k$")
plt.ylabel("max(Re($\sigma$))")
plt.title("max(Re($\sigma$)) vs. $k$, $\dot\gamma = {}$, $\\tau_a={}$".format(_gammadot, _tau_a))
plt.legend()
plt.show()
