from shear2D import ev
import numpy as np

_gammadot = 1
_tau = 1
_tau_a = 0.1

ks = np.arange(0,1,0.01)
for i in range(len(ks)):
    k = ks[i]
    mev = np.max(np.real(ev(k,_gammadot,_tau,_tau_a)))
    print(k, mev)