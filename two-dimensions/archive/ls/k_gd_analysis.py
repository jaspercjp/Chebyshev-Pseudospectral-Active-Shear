from shear2D import ev
import numpy as np
import matplotlib.pyplot as plt

_tau_a = 0.125
_tau = 1.0

_gds = [1,3,10,30,50]
kl, kr = 0, 10
k_step = 0.1
ks = np.arange(kl,kr,k_step)
mev = [None] * len(ks)

for _gd in _gds:
    for i in range(len(ks)):
        k = ks[i]
        mev[i] = np.max(np.real(ev(k, _gd, _tau, _tau_a)))
    plt.plot(ks, mev, label="$\dot\gamma$ = {}".format(_gd))

plt.xlabel("$k$")
plt.ylabel("max(Re($\sigma$))")
plt.title("max(Re($\sigma$)) vs. $k$, $\\tau= {}$, $\\tau_a={}$".format(_tau, _tau_a))
plt.legend()
plt.show()
