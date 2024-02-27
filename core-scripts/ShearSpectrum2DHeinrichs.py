import numpy as np
from scipy.linalg import eig
import numpy as np
from cheb import cheb
from HeinrichsBasis import HeinrichsBasis

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
	D,y = cheb(M)
	D2 = D@D 
	D3 = D2@D
	D4 = D2@D2

	# Take out the first and last rows 
	# D2[0,:]=0; D2[:,0]=0; D2[M-1,:]=0; D2[:,M-1]=0
	# D3[0,:]=0; D3[:,0]=0; D3[M-1,:]=0; D3[:,M-1]=0
	# D4[0,:]=0; D4[:,0]=0; D4[M-1,:]=0; D4[:,M-1]=0

	# Switch to a Heinrich basis (1-x^2)^2 * Q(x) instead
	S = np.zeros(M)
	S_p = np.zeros(M)
	S[1:M-1] = 1 / (1 - y[1:M-1]**2)**2
	S_p[1:M-1] = 1 / (1 - y[1:M-1]**2)
	I = np.identity(M)
	# D1_H0 = 4*y*(-1+y**2)*I
	# D1_H1 = ((-1+y**2)**2 * D1.T).T
	# D1_H = (D1_H0 + D1_H1) * S

	# Heinrichs D2
	# D2_H0 = -2*I 
	# D2_H1 = -4*(y*D.T).T
	# D2_H2 = ((1-y**2) * D2.T).T
	# D2_H = (D2_H0 + D2_H1 + D2_H2) * S_p
	# D2_H = D2_H[1:M-1,1:M-1]

	# Use Heinrichs basis to approximate the 4-th derivative
	# D4_aux = ((1-y**2)*D4.T).T
	# D3_aux = 8*(y*((D3).T)).T 
	# D2_aux = 12*D2
	D4_H0 = 24*I 
	D4_H1 = 96*(y*D.T).T
	D4_H2 = 24*((-1+3*y**2)*D2.T).T
	D4_H3 = 16*(y*(-1+y**2)*D3.T).T
	D4_H4 = (((-1+y**2)**2) * D4.T).T
	D4_H = (D4_H0 + D4_H1 + D4_H2 + D4_H3 + D4_H4) * S


	D1 = D[1:M-1, 1:M-1]; D2=D2[1:M-1,1:M-1]; D4=D4_H[1:M-1,1:M-1]

	# Variable layout
	# psi[0:M] Qxx[M:2*M] Qxy[2*M:3*M] 
	MM = M-2
	Rpsi  = slice(0*MM,1*MM)
	RQxx  = slice(1*MM,2*MM)
	RQxy  = slice(2*MM,3*MM)

	LHS = np.zeros((3*MM,3*MM),dtype='D')
	
	II = np.identity(MM)
	y=y[1:M-1]

	# Stokes equation
	LHS[Rpsi, Rpsi] = (k**4*II - 2*(k**2)*D2 + D4)/tau
	LHS[Rpsi,RQxx] = -2j*a*k*D1/_eta
	LHS[Rpsi, RQxy] = -(a*(k**2)*II + a*D2)/_eta

	## Qxx equation
	LHS[RQxx,Rpsi] = (_tmp_const*(gd*tau) * II\
							- 2j*k*_llambda*D1 - (_llambda*gd*tau/aux_const)*D2)
	LHS[RQxx,RQxx] = (1 + _ell_over_W_squared*k*k + 1j*k*y*gd*tau)*II - _ell_over_W_squared*D2
	LHS[RQxx,RQxy] = -gd*tau*II

	## Qxy equation
	LHS[RQxy,Rpsi] = (-_tmp_const*(1 + 2*(gd*tau)**2)*II)\
							- _llambda / aux_const * D2
	LHS[RQxy,RQxx] = gd*tau*II
	LHS[RQxy,RQxy] = (1 + _ell_over_W_squared*k*k + 1j*k*y*gd*tau)*II - _ell_over_W_squared*D2

	# Specifying the right hand side of the eigenvalue problem
	RHS = np.zeros((3*MM,3*MM),dtype='D')
	RHS[RQxx,RQxx] = -tau*II
	RHS[RQxy,RQxy] = -tau*II

	## Boundary conditions
	# LHS[0]     = np.zeros(3*MM,dtype='D') # Psi vanishes at the boundaries
	# LHS[1]     = np.zeros(3*MM,dtype='D') # and dy(Psi) (?) vanishes at the boundaries
	# LHS[MM-2]   = np.zeros(3*MM,dtype='D') # how does this code account for no-slip?
	# LHS[MM-1]   = np.zeros(3*MM,dtype='D')

	# LHS[0,0]      = 1.0
	# LHS[1,Rpsi]   = D1[0]
	# LHS[MM-2,Rpsi] = D1[MM-1]
	# LHS[MM-1,MM-1]  = 1.0

	LHS[MM]     = np.zeros(3*MM,dtype='D') # dy(Qxx) vanishes at the boundaries
	LHS[2*MM-1] = np.zeros(3*MM,dtype='D')
	LHS[MM,RQxx]     = D[0,1:M-1]
	LHS[2*MM-1,RQxx] = D[-1,1:M-1]

	LHS[2*MM]   = np.zeros(3*MM,dtype='D') # dy(Qxy) vanishes at the boundaries
	LHS[3*MM-1] = np.zeros(3*MM,dtype='D')
	LHS[2*MM,RQxy]   = D[0,1:M-1]
	LHS[3*MM-1,RQxy] = D[-1,1:M-1]


	# RHS[0]     = np.zeros(3*MM,dtype='D')
	# RHS[1]     = np.zeros(3*MM,dtype='D')
	# RHS[MM-2]   = np.zeros(3*MM,dtype='D')
	# RHS[MM-1]   = np.zeros(3*MM,dtype='D')
	RHS[MM]     = np.zeros(3*MM,dtype='D')
	RHS[2*MM-1] = np.zeros(3*MM,dtype='D')
	RHS[2*MM]   = np.zeros(3*MM,dtype='D')
	RHS[3*MM-1] = np.zeros(3*MM,dtype='D')


	evs, emodes = eig(LHS,RHS,left=0,right=1)
	print("LHS Cond = ", np.linalg.cond(LHS))
	print("RHS Cond = ", np.linalg.cond(RHS))
	
	# clean the eigenvalue list of infinities
	finite_idx = np.where(np.isfinite(evs))[0]
	clean_eig_list = evs[finite_idx]
	clean_modes_list = emodes[:,finite_idx]

	return (y, clean_eig_list, clean_modes_list)