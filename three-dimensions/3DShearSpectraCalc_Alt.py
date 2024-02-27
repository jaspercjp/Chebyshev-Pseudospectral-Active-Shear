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

# kx = 1
kz = 0
# These grid values were used by Wan to do all her calculations
kxl,kxr,kxn = 0.0, 1.0, 11
kxs = np.linspace(kxl,kxr,kxn)
al, ar, an = 0.0, 2.5, 26
acts = np.linspace(al, ar, an)
SAVE_PATH = os.path.join("data", f"3D-shear-new-gd0-kz0")
if not os.path.isdir(SAVE_PATH):
    os.system("mkdir " + SAVE_PATH)
np_save_file(f"{SAVE_PATH}", "kxs", kxs)
np_save_file(f"{SAVE_PATH}", "acts", acts)

for i in tqdm(range(len(kxs))):
    for j in range(len(acts)):
        evals_low, _ = ShearSpectrum3D.spectrum(kx=kxs[i],kz=kz,gd=0,a=acts[j],M=50)
        evals_high, _ = ShearSpectrum3D.spectrum(kx=kxs[i],kz=kz,gd=0,a=acts[j],M=70)
        np_save_file(os.path.join(f"{SAVE_PATH}", "M-50"), f"kx-{kxs[i]}-a-{acts[j]}", evals_low)
        np_save_file(os.path.join(f"{SAVE_PATH}", "M-70"), f"kx-{kxs[i]}-a-{acts[j]}", evals_high)