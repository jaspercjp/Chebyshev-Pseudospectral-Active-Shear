from shear2D_tau import ev_tau
import numpy as np
import copy

k=1
_tau=1.0

M,N = 10, 7
step = 0.1
_gds = np.arange(0.01,M,step)
_as = np.arange(0,4,step)
tau_a_step = 0.5
last_tau_a = 1
for i in range(len(_gds)):
	tau_a=last_tau_a
	mev = np.max(np.real(ev(k,_gds[i],_tau,tau_a)))
	while np.real(mev)<0:
		tau_a = tau_a / (1 + step*tau_a_step)
		mev = max_ev(k,_gds[i],_tau,tau_a)	
	print("change in 1/tau_a= {:.4f}".format(1/tau_a - 1/last_tau_a))
	last_tau_a = tau_a
	#print("unstable at k={}, tBar={:.4f}, 1/tau_a={:.4f}, fastest growth rate:{:.4f} @ freq={:.4f}"\
	#	.format(k,_gds[i]*_tau,1/(tau_a),np.real(mev),np.imag(mev)))
	
	#print("{} out of {}")
#print(stab_arr)
		
