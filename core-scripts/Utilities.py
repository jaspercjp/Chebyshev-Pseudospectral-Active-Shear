import numpy as np
import ShearSpectrum2D

def prune_evs(data_low, data_high, threshold=25, eps=0.001):
    """
    Takes in two lists of complex numbers computed from the Chebyshev spectral method
    with different resolutions, and outputs a "pruned version"
    computed from these two lists. The pruning is done by sorting the two
    lists by their real parts, and matching the very first entry -- then taking the first
    25 values.
    """
    assert len(data_low)>=2*threshold, "It is recommended that the threshold number of values to take is at most around half of the input size"
    ptr_1 = 0
    ptr_2 = 0
    sort_idx_low = np.argsort(-np.real(data_low))
    sort_idx_high = np.argsort(-np.real(data_high))
    sorted_low = data_low[sort_idx_low]
    sorted_high = data_high[sort_idx_high]
    while not abs(np.real(sorted_low[ptr_1]) - np.real(sorted_high[ptr_2])) < eps:
        if np.real(sorted_low[ptr_1]) > np.real(sorted_high[ptr_2]):
            ptr_1 += 1
        elif np.real(sorted_low[ptr_1]) < np.real(sorted_high[ptr_2]):
            ptr_2 += 1
    # cleaned low might not really be necessary
    # cleaned_low = sorted_low[ptr_1:threshold]
    cleaned_high = sorted_high[ptr_2:threshold]
    return cleaned_high

def clean_spec_2D(k,gd,a, tau=1, ell_over_W_squared=0.01, M_low=50, M_high=100):
    data_low = np.array(ShearSpectrum2D.spectrum(k,gd,tau,a,_ell_over_W_squared=ell_over_W_squared, M=M_low), dtype=np.complex128)
    data_high = np.array(ShearSpectrum2D.spectrum(k,gd,tau,a,_ell_over_W_squared=ell_over_W_squared, M=M_high), dtype=np.complex128)
    return prune_evs(data_low, data_high)

def max_re(zs):
    acc = -float('inf')
    index = -1
    for  i in range(len(zs)):
        z = zs[i]
        if z.real > acc.real:
            acc = z
            index = i
    return acc,index