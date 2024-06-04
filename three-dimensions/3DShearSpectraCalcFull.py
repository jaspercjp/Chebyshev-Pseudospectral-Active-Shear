"""
    Script calculating spectra of the active gel at zero shear 
    for both 2D and 3D using a fine scale for the activity. This is 
    to see if one system goes unstable before the other.
"""

import sys
import os
sys.path.append("../core-scripts")
import ShearSpectrum3DFull as S3D
from Utilities import prune_evs, np_save_file
import numpy as np
from tqdm import tqdm

kz = 0.9
# These grid values were used by Wan to do all her calculations
gdl,gdr,gdn = 0.0, 1.0, 11
gd = np.linspace(gdl, gdr, gdn)
al, ar, an = 0.0, 2.5, 26
# al, ar, an = -5.0, -2.5, 26
a = np.linspace(al, ar, an)

# Do the same calculation over different k values
kxl, kxr, kxn = 0, 4, 7
kx = np.linspace(kxl, kxr, kxn)

LOW_RES = 170
HIGH_RES = 190
NUM_TO_KEEP = 100 # how many eigenvalues to keep within each calculated spectrum

SAVE_PATH = os.path.join("data", f"3D-shear-FULL-kx_{kxl}-{kxr}_kz{kz}-M-{LOW_RES}-{HIGH_RES}")
if not os.path.isdir(SAVE_PATH):
	os.system("mkdir " + SAVE_PATH)
	
# Save the discretized grid 
np_save_file(f"{SAVE_PATH}", "gds", gd)
np_save_file(f"{SAVE_PATH}", "acts", a)
np_save_file(f"{SAVE_PATH}", "kxs", kx)

# AXIS-0 <=> kx | 1 <=> gd | 2 <=> a | 3 <=> LOW/HIGH res | 4 <=> manual threshold on how many eigenvalues to keep
spectra_mat = np.zeros((kxn, gdn, an, 2, NUM_TO_KEEP), dtype=np.complex128)

for i in range(gdn):
	print(" ========================== // ===========================")
	for j in range(an):
		for k in range(kxn):
			evals_low, _ = S3D.spectrum(kx=kx[k],kz=kz,gd=gd[i],a=a[j],M=LOW_RES)
			evals_high, _ = S3D.spectrum(kx=kx[k],kz=kz,gd=gd[i],a=a[j],M=HIGH_RES)
			evals_low = evals_low[np.argsort(-evals_low.real)]
			evals_high = evals_high[np.argsort(-evals_high.real)]
			spectra_mat[k,i,j,0,:] = evals_low[:NUM_TO_KEEP]
			spectra_mat[k,i,j,1,:] = evals_high[:NUM_TO_KEEP]
			print(f"Done. gd:{i+1}/{gdn}, a:{j+1}/{an}, k:{k+1}/{kxn}.")
np_save_file(f"{SAVE_PATH}", "spectra-mat", spectra_mat)
