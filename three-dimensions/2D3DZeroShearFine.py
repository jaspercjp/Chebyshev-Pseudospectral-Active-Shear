"""
    Script calculating spectra of the active gel at zero shear 
    for both 2D and 3D using a fine scale for the activity. This is 
    to see if one system goes unstable before the other.
"""

import sys
sys.path.append("../core-scripts")
import zeroShearSpectrum2D
import zeroShearSpectrum3D
from Utilities import prune_evs
import numpy as np
from tqdm import tqdm

def compute_zero_shear_specs(k,a,x=0.01,M_low=50,M_high=100):
    low_res_2D, _ = zeroShearSpectrum2D.spectrum(k,a,_ell_over_W_squared=x,M=M_low)
    high_res_2D, _ = zeroShearSpectrum2D.spectrum(k,a,_ell_over_W_squared=x,M=M_high)
    low_res_3D, _ = zeroShearSpectrum3D.spectrum(k,a,_ell_over_W_squared=x,M=M_low)
    high_res_3D, _ = zeroShearSpectrum3D.spectrum(k,a,_ell_over_W_squared=x,M=M_high)
    low_res_2D = np.array(low_res_2D, dtype=np.complex128)
    high_res_2D = np.array(high_res_2D, dtype=np.complex128)
    low_res_3D = np.array(low_res_3D, dtype=np.complex128)
    high_res_3D = np.array(high_res_3D, dtype=np.complex128)
    pruned_2D = prune_evs(low_res_2D, high_res_2D)
    pruned_3D = prune_evs(low_res_3D, high_res_3D)
    data = np.zeros((3, 25), dtype=np.complex128) # 25 is hard-coded limit 
    sort_idx = np.argsort(-np.real(low_res_2D))
    sorted_low_res_2D = low_res_2D[sort_idx]
    data[0,:] = sorted_low_res_2D[0:25]
    data[1,:] = pruned_2D 
    data[2,:] = pruned_3D
    return data

ks = np.linspace(0,4,25)
acts = np.linspace(0,1.5,50)
np.save("data/2D-3D-zero-shear-comp-fine/ks", ks)
np.save("data/2D-3D-zero-shear-comp-fine/acts", acts)

for i in tqdm(range(len(ks))):
    for j in range(len(acts)):
        data = compute_zero_shear_specs(ks[i], acts[j])
        # np.save(f"data/2D-3D-zero-shear-comp-fine/k{i}-a{j}", data)