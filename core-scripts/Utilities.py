import numpy as np
import ShearSpectrum2D
import matplotlib.pyplot as plt
import os

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
        if (ptr_1>=len(sorted_low) or ptr_2>=len(sorted_high)):
               return np.array([])
    # cleaned low might not really be necessary

    # cleaned_low = sorted_low[ptr_1:threshold]
    cleaned_high = sorted_high[ptr_2:ptr_2+threshold]
    return cleaned_high, ptr_1, ptr_2

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

def plot_modes(idx, ygl, evecs, M):
	""" Plots the mode at index "idx" on the coordinates "ygl" """
	fig, axs = plt.subplots(3, 3)
	Vx = np.reshape(evecs[0:M, idx], -1)
	axs[0,0].plot(ygl, np.real(Vx))
	axs[0,0].plot(ygl, np.imag(Vx))
	axs[0,0].title.set_text('$V_x$')

	Vy = np.reshape(evecs[M:2*M,idx], -1)
	axs[0,1].plot(ygl, np.real(Vy))
	axs[0,1].plot(ygl, np.imag(Vy))
	axs[0,1].title.set_text('$V_y$')

	Vz = np.reshape(evecs[2*M:3*M,idx], -1)
	axs[0,2].plot(ygl, np.real(Vz))
	axs[0,2].plot(ygl, np.imag(Vz))
	axs[0,2].title.set_text('$V_z$')

	Qxx = np.reshape(evecs[3*M:4*M,idx], -1)
	axs[1,0].plot(ygl, np.real(Qxx))
	axs[1,0].plot(ygl, np.imag(Qxx))
	axs[1,0].title.set_text('$Q_{xx}$')

	Qxy = np.reshape(evecs[4*M:5*M,idx], -1)
	axs[1,1].plot(ygl, np.real(Qxy))
	axs[1,1].plot(ygl, np.imag(Qxy))
	axs[1,1].title.set_text('$Q_{xy}$')

	Qxz = np.reshape(evecs[5*M:6*M,idx], -1)
	axs[1,2].plot(ygl, np.real(Qxz))
	axs[1,2].plot(ygl, np.imag(Qxz))
	axs[1,2].title.set_text('$Q_{xz}$')

	Qyz = np.reshape(evecs[6*M:7*M,idx], -1)
	axs[2,0].plot(ygl, np.real(Qyz))
	axs[2,0].plot(ygl, np.imag(Qyz))
	axs[2,0].title.set_text('$Q_{yz}$')

	Qzz = np.reshape(evecs[7*M:8*M,idx], -1)
	axs[2,1].plot(ygl, np.real(Qzz))
	axs[2,1].plot(ygl, np.imag(Qzz))
	axs[2,1].title.set_text('$Q_{zz}$')

	fig.tight_layout()

	for ax in axs.flat:
		ax.set(xlabel='$y$')

	return fig

def np_save_file(parent_path, filename, obj):
    if not os.path.isdir(parent_path):
          os.system("mkdir " + parent_path) 
    np.save(os.path.join(parent_path, filename), obj)
    
def power_iteration(A, num_iterations: int):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
	b_k = np.random.rand(A.shape[1])

	for _ in range(num_iterations):
		# calculate the matrix-by-vector product Ab
		b_k1 = np.dot(A, b_k)

		# calculate the norm
		b_k1_norm = np.linalg.norm(b_k1)

		# re normalize the vector
		b_k = b_k1 / b_k1_norm
          
	ev_k = np.dot(A, b_k) / np.norm(b_k)
	return (ev_k, b_k)

def calc_scaled_diff(low_evals, high_evals):
	low_evals = low_evals[np.argsort(-np.real(low_evals))]
	high_evals = high_evals[np.argsort(-np.real(high_evals))]

	d_ordinal = np.zeros(len(low_evals)-1)
	d_nearest = np.zeros(len(low_evals)-1)

	for i in range(len(low_evals)-1):
		if i == 0:
			sigma_low = np.abs(low_evals[i] - low_evals[i+1])
			sigma_high = np.abs(high_evals[i] - high_evals[i+1])
		else:
			sigma_low = (np.abs(low_evals[i]-low_evals[i-1]) + np.abs(low_evals[i+1]-low_evals[i])) / 2
			sigma_high = (np.abs(high_evals[i]-high_evals[i-1]) + np.abs(high_evals[i+1]-high_evals[i])) / 2
		
		aux_diff = np.abs(low_evals[i] - high_evals)
		delta_nearest = np.min(aux_diff) / np.min([np.abs(low_evals[i]), sigma_low])
		delta_ordinal = aux_diff[i] / np.min([np.abs(low_evals[i]), sigma_low])

		d_ordinal[i] = delta_ordinal
		d_nearest[i] = delta_nearest
	
	return d_ordinal, d_nearest

def plot_scaled_diff(low_evals, high_evals):
	d_ordinal, d_nearest = calc_scaled_diff(low_evals, high_evals)

	fig = plt.figure()
	# print(len(np.arange(len(low_evals)-1)), len(1/(d_ordinal+1e-7)))
	plt.scatter(np.arange(len(low_evals)-1), 1/(d_ordinal+1e-7), marker='x')
	plt.scatter(np.arange(len(low_evals)-1), 1/(d_nearest+1e-7), marker='o', facecolors='none', edgecolors='y')
	plt.ylabel("$1/\delta$")
	plt.xlabel("mode number")
	plt.title("Plot of Scaled Differences, 'Nearest' and 'Ordinal' ")
	plt.legend(["Ordinal", "Nearest"])
	ax = plt.gca()
	ax.set_yscale('log')

	return (fig, d_ordinal, d_nearest)

def extract_crit_a(ev_arr, a_arr):
	gdn, an = ev_arr.shape
	crit_a = np.zeros(gdn)
	for i in range(gdn):
		for j in range(an-1):
			ev = ev_arr[i,j]
			next_ev = ev_arr[i,j+1]
			if np.real(ev) * np.real(next_ev) < 0:
				# interpolate between the sign change for critical activity
				crit_a[i] = (a_arr[j] + a_arr[j+1]) / 2
				break
	return crit_a

def extract_max_re_sig(ev_arr_k):
	kn, gdn, an, _ = ev_arr_k.shape
	ev_arr_temp = np.zeros((gdn, an, kn), dtype=np.complex128)
	ev_arr = np.zeros((gdn, an), dtype=np.complex128)
	# sort each spectra first 
	for i in range(gdn):
		for j in range(an):
			for k in range(kn):
				evs = ev_arr_k[k,i,j,:]
				sorted_ev_k = evs[np.argsort(-evs.real)]
				ev_arr_temp[i,j,k] = sorted_ev_k[0]
	for i in range(gdn):
		for j in range(an):
			ev_arr[i,j] = max_re(ev_arr_temp[i,j,:])[0]
	return ev_arr