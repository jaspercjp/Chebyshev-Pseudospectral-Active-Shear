"""
    Script calculating spectra of the active gel at zero shear 
    for both 2D and 3D using a fine scale for the activity. This is 
    to see if one system goes unstable before the other.
"""

import sys
import os
sys.path.append("../core-scripts")
import ShearSpectrum3D
from Utilities import prune_evs, np_save_file
import numpy as np
from tqdm import tqdm

kx = 1
kz = 0.0
# These grid values were used by Wan to do all her calculations
gdl,gdr,gdn = 0.0, 1.0, 2
gds = np.linspace(gdl, gdr, gdn)
al, ar, an = 0.0, 2.5, 2
acts = np.linspace(al, ar, an)

# Do the same calculation over many many different k values
kxl, kxr, kxn = 0, 20, 2
kxs = np.linspace(kxl, kxr, kxn)

SAVE_PATH = os.path.join("data", f"3D-shear-new-kx{kx}-kz{kz}")
if not os.path.isdir(SAVE_PATH):
	os.system("mkdir " + SAVE_PATH)

# Save the discretized grid 
np_save_file(f"{SAVE_PATH}", "gds", gds)
np_save_file(f"{SAVE_PATH}", "acts", acts)
np_save_file(f"{SAVE_PATH}", "kxs", kxs)

LOW_RES = 50
HIGH_RES = 70
NUM_TO_KEEP = 120 # how many eigenvalues to keep within each calculated spectrum

# AXIS-0 <=> kx | 1 <=> gd | 2 <=> a | 3 <=> LOW/HIGH res | 4 <=> how many eigenvalues we want to keep
spectra_mat = np.zeros((kxn, gdn, an, 2, NUM_TO_KEEP))
for k in tqdm(range(kxn)):
	for i in tqdm(range(gdn)):
		for j in range(an):
			evals_low, _ = ShearSpectrum3D.spectrum(kx=kx,kz=kz,gd=gds[i],a=acts[j],M=LOW_RES)
			evals_high, _ = ShearSpectrum3D.spectrum(kx=kx,kz=kz,gd=gds[i],a=acts[j],M=HIGH_RES)
			evals_low = evals_low[np.argsort(-evals_low.real)]
			evals_high = evals_high[np.argsort(-evals_high.real)]
			spectra_mat[k,i,j,0,:] = evals_low[:NUM_TO_KEEP]
			spectra_mat[k,i,j,1,:] = evals_high[:NUM_TO_KEEP]
			print(f"Done. k:{k+1}/{kxn}, gd:{i+1}/{gdn}, a:{j+1}/{an}")
np_save_file(f"{SAVE_PATH}", "spectra-mat", spectra_mat)