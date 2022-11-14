"""
Evaluate spectra on a grid of gammadot*tau and k
"""

from shear2D import max_ev_kg_grid
import numpy as np
import matplotlib.pyplot as plt

tau, tau_a = 1, 0.05
gdl, gdr, gd_step = 1, 5, 0.5
kl, kr, k_step = 1, 5, 0.5

ks = np.arange(kl, kr, k_step)
gds = np.arange(gdl, gdr, gd_step)
kv, gv = np.meshgrid(ks, gds)

# denote the matrix sizes as NxM
# Nk, Mk = len(kv), len(ks[0])
# Ng, Mg = len(gds), len(gds[0])

mat = max_ev_kg_grid(ks,gds,tau,tau_a)
print(mat)
plt.plot(kv.flat, gv.flat, ".")