import numpy as np
from scipy.linalg import eig
import numpy as np
from cheb import cheb

##------------------------------------
def spectrum(k,gd,tau,a,_ell_over_W_squared=0.01, M=50):
	"""Takes in a set of parameters and returns the spectrum that 
	corresponds to these parameter values.

	Args:
		k (float): wavenumber
		_gammadot (float): external shear rate
		tau (float): liquid crystal relaxation time
		taua (float): activity time scale

	Returns:
		list[complx numbers]: a (cleaned of any infinities) list of eigenvalues
	"""
	# M is the resolution

	# parameters 
	_llambda = 1.0
	_eta = 1.0
	aux_const = 1 + (gd*tau)**2
	_tmp_const = k*k*_llambda / aux_const 

	##------------------------------------

	II = np.identity(M,dtype='d')

	# Build Chebyshev differention matrix and the grid points on [-1, 1]
	D1, ygl = cheb(M)

	# let's just use [-1, 1] for simpliciy...
	# The factor 2 takes care of the domain being from -1/2 to 1/2
	# D1 = 2*D1

	D2 = np.dot(D1,D1)

	# Variable layout
	# psi[0:M] Qxx[M:2*M] Qxy[2*M:3*M] 
	Rpsi  = slice(0*M,1*M)
	RQxx  = slice(1*M,2*M)
	RQxy  = slice(2*M,3*M)

	LHS = np.zeros((3*M,3*M),dtype='D')

	# Stokes equation
	LHS[Rpsi, Rpsi] = (k**4*II - 2*(k**2)*D2 + np.dot(D2,D2))/tau
	LHS[Rpsi,RQxx] = -2j*a*k*D1/_eta
	LHS[Rpsi, RQxy] = -(a*(k**2)*II + a*D2)/_eta

	## Qxx equation
	LHS[RQxx,Rpsi] = (_tmp_const*(gd*tau) * II\
							- 2j*k*_llambda*D1 - (_llambda*gd*tau/aux_const)*D2)
	LHS[RQxx,RQxx] = (1 + _ell_over_W_squared*k*k + 1j*k*ygl*gd*tau)*II - _ell_over_W_squared*D2
	LHS[RQxx,RQxy] = -gd*tau*II

	## Qxy equation
	LHS[RQxy,Rpsi] = (-_tmp_const*(1 + 2*(gd*tau)**2)*II)\
							- _llambda / aux_const * D2
	LHS[RQxy,RQxx] = gd*tau*II
	LHS[RQxy,RQxy] = (1 + _ell_over_W_squared*k*k + 1j*k*ygl*gd*tau)*II - _ell_over_W_squared*D2

	RHS = np.zeros((3*M,3*M),dtype='D')
	RHS[RQxx,RQxx] = -tau*II
	RHS[RQxy,RQxy] = -tau*II

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


	evs, emodes = eig(LHS,RHS,left=0,right=1)
	print("LHS Cond = ", np.linalg.cond(LHS))
	print("RHS Cond = ", np.linalg.cond(RHS))
	
	# clean the eigenvalue list of infinities
	finite_idx = np.where(np.isfinite(evs))[0]
	clean_eig_list = evs[finite_idx]
	clean_modes_list = emodes[:,finite_idx]

	return (clean_eig_list, clean_modes_list)