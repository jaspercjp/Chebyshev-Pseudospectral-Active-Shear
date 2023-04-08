import numpy as np
from scipy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

##------------------------------------
def ev_tau(k,_tau,_a,_ell_over_W_squared=0.01, M=50):
	"""Takes in a set of parameters and returns the spectrum that 
	corresponds to these parameter values.

	Args:
		k (float): wavenumber
		_gammadot (float): external shear rate
		_tau (float): liquid crystal relaxation time
		_tau_a (float): activity time scale

	Returns:
		list[complx numbers]: a (cleaned of any infinities) list of eigenvalues
	"""
	# M is the resolution

	# parameters 
	_llambda = 1.0
	_eta = 1.0
	_aux_const = 1.0


	##------------------------------------
	mode_number = 94 #79
	do_plot_mode = 0

	##------------------------------------
	print("change detected")
	II = np.identity(M,dtype='d')

	cbar = np.ones(M,dtype='d')
	cbar[0] = 2.0
	cbar[M-1] = 2.0

	# Chebyshev grid points
	ygl = np.zeros(M,dtype='d')
	for m in range(M):
		ygl[m] = np.cos(np.pi*m/(M-1))
	# ygl = 1/2 * ygl

	# Build Chebyshev differentiation matrix
	D1 = np.zeros((M,M),dtype='d')
	for l in range(M):
		for j in range(M):
			if l != j:
				D1[l,j] = cbar[l]*((-1)**(l+j))/(cbar[j]*(ygl[l]-ygl[j]))

	for j in range(1,M-1):
		D1[j,j] = -0.5*ygl[j]/(1.0-ygl[j]*ygl[j])

	D1[0,0] = (2.0*(M-1)*(M-1)+1.0)/6.0
	D1[M-1,M-1] = -D1[0,0]

	# The factor 2 takes care of the domain being from -1/2 to 1/2. 
	# This is equivalent to making the change of variable x -> x/2
	# D1 = 2*D1

	D2 = np.dot(D1,D1)

	## Auxiliary matrices

	Lmin = D2 - k*k*II 
	Lplus = D2 + k*k*II 


	## LHS

	## Variable layout
	## psi[0:M] Qxx[M:2*M] Qxy[2*M:3*M] 

	Rpsi  = slice(0*M,1*M)
	RQxx  = slice(1*M,2*M)
	RQxy  = slice(2*M,3*M)


	LHS = np.zeros((3*M,3*M),dtype='D')

	# Stokes equation
	LHS[Rpsi, Rpsi] = np.dot(D2,D2)/_tau
	LHS[Rpsi,RQxx] = 0*II
	LHS[Rpsi, RQxy] = -_a*D2/_eta

	## Qxx equation
	LHS[RQxx,Rpsi] = 0*II
	LHS[RQxx,RQxx] = II - _ell_over_W_squared*D2
	LHS[RQxx,RQxy] = 0*II

	## Qxy equation
	LHS[RQxy,Rpsi] = -_llambda / _aux_const * D2
	LHS[RQxy,RQxx] = 0*II
	LHS[RQxy,RQxy] = II - _ell_over_W_squared*D2

	RHS = np.zeros((3*M,3*M),dtype='D')
	RHS[RQxx,RQxx] = -_tau*II
	RHS[RQxy,RQxy] = -_tau*II

	## Boundary conditions

	LHS[0]     = np.zeros(3*M,dtype='D') # Psi vanishes at the boundaries
	LHS[1]     = np.zeros(3*M,dtype='D') # and dy(Psi) (?) vanishes at the boundaries
	LHS[M-2]   = np.zeros(3*M,dtype='D') # how does this code account for no-slip?
	LHS[M-1]   = np.zeros(3*M,dtype='D')

	LHS[0,0]      = 1.0
	LHS[1,Rpsi]   = D1[0]
	LHS[M-2,Rpsi] = D1[M-1]
	LHS[M-1,M-1]  = 1.0

	LHS[M]     = np.zeros(3*M,dtype='D') # dy(Qxx) vanishes at the boundaries
	LHS[2*M-1] = np.zeros(3*M,dtype='D')

	LHS[M,RQxx]     = D1[0]
	LHS[2*M-1,RQxx] = D1[M-1]

	LHS[2*M]   = np.zeros(3*M,dtype='D') # dy(Qxy) vanishes at the boundaries
	LHS[3*M-1] = np.zeros(3*M,dtype='D')

	LHS[2*M,RQxy]   = D1[0]
	LHS[3*M-1,RQxy] = D1[M-1]


	RHS[0]     = np.zeros(3*M,dtype='D')
	RHS[1]     = np.zeros(3*M,dtype='D')
	RHS[M-2]   = np.zeros(3*M,dtype='D')
	RHS[M-1]   = np.zeros(3*M,dtype='D')
	RHS[M]     = np.zeros(3*M,dtype='D')
	RHS[2*M-1] = np.zeros(3*M,dtype='D')
	RHS[2*M]   = np.zeros(3*M,dtype='D')
	RHS[3*M-1] = np.zeros(3*M,dtype='D')


	_spec = eig(LHS,RHS,left=0,right=1)

	_eig_list = _spec[0]
	_modes_list = _spec[1]

	_clean_eig_list = list(filter(lambda ev: np.isfinite(ev), _eig_list))
	return _clean_eig_list