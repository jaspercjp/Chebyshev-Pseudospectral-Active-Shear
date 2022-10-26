from shear2D import max_ev
import numpy as np
import copy

def gen_seq(start,step,n):
	z0=start
	zs = np.zeros(n)
	zs[0]=z0
	for i in range(1,len(zs)):
		zs[i] = zs[i-1] / (1 + step * zs[i-1])
	return zs

k=1
_tau=1.0

M,N = 10, 7
step = 0.1
#stab_arr = np.zeros((int(M/step),int(N/step)),dtype='f')
_gds = np.arange(0.01,M,step)
_tau_as = gen_seq(5,0.2,30)
tau_a_step = 0.5
last_tau_a = 1
for i in range(len(_gds)):
	tau_a=last_tau_a
	mev = max_ev(k,_gds[i],_tau,tau_a)
	while np.real(mev)<0:
		tau_a = tau_a / (1 + step*tau_a_step)
		mev = max_ev(k,_gds[i],_tau,tau_a)	
	print("change in 1/tau_a= {:.4f}".format(1/tau_a - 1/last_tau_a))
	last_tau_a = tau_a
	#print("unstable at k={}, tBar={:.4f}, 1/tau_a={:.4f}, fastest growth rate:{:.4f} @ freq={:.4f}"\
	#	.format(k,_gds[i]*_tau,1/(tau_a),np.real(mev),np.imag(mev)))
	
	#print("{} out of {}")
#print(stab_arr)
		
