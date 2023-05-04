import numpy as np
from scipy.linalg import eig
import numpy as np
from cheb import cheb

##------------------------------------
def spectrum(kx,a,_ell_over_W_squared=0.01, M=50):
	"""Takes in a set of parameters and returns the spectrum that 
	corresponds to these parameter values.

	Args:
		k (float): wavenumber
		_gammadot (float): external shear rate

	Returns:
		list[complx numbers]: a (cleaned of any infinities) list of eigenvalues
	"""
	# M is the resolution

	# parameters 
	llambda = 1.0
	eta = 1.0
	tau = 1.0

	##------------------------------------

	II = np.identity(M,dtype='d')

	# Build Chebyshev differentiation matrix and the grid points on [-1, 1]
	D1, ygl = cheb(M)
	D2 = np.dot(D1,D1)
	D3 = np.dot(D2,D1)

	# Variable layout
	# psi[0:M] Qxx[M:2*M] Qxy[2*M:3*M] 
	RVx  = slice(0*M,1*M)
	RVy = slice(1*M, 2*M)
	RVz = slice(2*M, 3*M)
	RQxx  = slice(3*M,4*M)
	RQxy  = slice(4*M,5*M)
	RQxz = slice(5*M, 6*M)
	RQyz = slice(6*M, 7*M)
	RQzz = slice(7*M, 8*M)

	LHS = np.zeros((8*M,8*M),dtype='D')

    # Vx equation
	LHS[RVx, RVz] = (kx**2)*eta*D1 - eta*D3
	LHS[RVx, RQxz] = 1j*a*kx*tau*D1 
	LHS[RVx, RQyz] = a*tau*D2
	# Vy equation
	LHS[RVy, RVz] = -1j*(kx**3)*eta*II + 1j*kx*eta*D2 
	LHS[RVy, RQxz] = a*(kx**2)*tau*II
	LHS[RVy, RQyz] = -1j*a*kx*tau*D1 
	# Vz equation
	LHS[RVz,RVy] = 1j*(kx**3)*eta*II - 1j*kx*eta*D2 
	LHS[RVz,RQxy] = -a*(kx**2)*tau*II - a*tau*D2 
	LHS[RVz,RQxx] = -2j*a*kx*tau*D1 
	LHS[RVz,RQzz] = -1j*a*kx*tau*D1 
	LHS[RVz,RVx] = -(kx**2)*eta*D1 + eta*D3

	## Qxx equation
	LHS[RQxx,RQxx] = II + _ell_over_W_squared*(kx**2)*II - _ell_over_W_squared*D2 
	LHS[RQxx,RVx] = -2j*kx*II

	## Qxy equation
	LHS[RQxy,RQxy] = II + _ell_over_W_squared*(kx**2)*II - _ell_over_W_squared*D2
	LHS[RQxy,RVy] = -1j*kx*II 
	LHS[RQxy,RVx] = -D1
	## Qxz equation
	LHS[RQxz, RQxz] = II + _ell_over_W_squared*(kx**2)*II - _ell_over_W_squared*D2 
	LHS[RQxz, RVz] = -1j*kx*II
	## Qyz equation
	LHS[RQyz, RQyz] = II + _ell_over_W_squared*(kx**2)*II - _ell_over_W_squared*D2 
	LHS[RQyz, RVz] = -D1
	## Qzz equation. This equation seems to be decoupled from all the other quantites! Is this right?
	LHS[RQzz, RQzz] = II + _ell_over_W_squared*(kx**2)*II - _ell_over_W_squared*D2 
	

	RHS = np.zeros((8*M,8*M),dtype='D')
	RHS[RQxx,RQxx] = -tau*II
	RHS[RQxy,RQxy] = -tau*II
	RHS[RQxz,RQxz] = -tau*II
	RHS[RQyz,RQyz] = -tau*II
	RHS[RQzz,RQzz] = -tau*II 

	## Boundary conditions
    # Setup zeros for Vx's boundary conditions
	LHS[0]     = np.zeros(8*M,dtype='D')
	# For now let us not use the BCs on derivatives!
	# LHS[1]     = np.zeros(3*M,dtype='D')
	# LHS[M-2]   = np.zeros(3*M,dtype='D')
	LHS[M-1]   = np.zeros(8*M,dtype='D')

    # Vx vanishes at boundary
	LHS[0,0]      = 1.0
	# LHS[1,Rpsi]   = D1[0]
	# LHS[M-2,Rpsi] = D1[M-1]
	LHS[M-1,M-1]  = 1.0

	LHS[M]     = np.zeros(8*M,dtype='D') # Vy vanishes at boundary
	LHS[2*M-1] = np.zeros(8*M,dtype='D')
	LHS[M,M]     = 1.0
	LHS[2*M-1,2*M-1] = 1.0

	LHS[2*M]   = np.zeros(8*M,dtype='D') # Vz vanishes at the boundary
	LHS[3*M-1] = np.zeros(8*M,dtype='D')
	LHS[2*M,2*M] = 1.0
	LHS[3*M-1,3*M-1] = 1.0
	
	LHS[3*M] = np.zeros(8*M,dtype='D') # dy(Qxx) vanishes 
	LHS[4*M-1] = np.zeros(8*M,dtype='D')
	LHS[3*M, RQxx] = D1[0]
	LHS[4*M-1, RQxx] = D1[M-1]
	
	LHS[4*M] = np.zeros(8*M, dtype='D') # dy(Qxy) vanishes at the boundary
	LHS[5*M-1] = np.zeros(8*M, dtype='D')
	LHS[4*M, RQxy] = D1[0]
	LHS[5*M-1, RQxy] = D1[M-1]
    
	LHS[5*M] = np.zeros(8*M, dtype='D') # dy(Qxz) vanishes at the boundary
	LHS[6*M-1] = np.zeros(8*M,dtype='D')
	LHS[5*M,RQxz] = D1[0]
	LHS[6*M-1,RQxz] = D1[M-1]
	
	LHS[6*M] = np.zeros(8*M,dtype='D') # dy(Qyz) vanishes
	LHS[7*M-1] = np.zeros(8*M,dtype='D')
	LHS[6*M,RQyz] = D1[0]
	LHS[7*M-1,RQyz] = D1[M-1]
	
	LHS[7*M] = np.zeros(8*M, dtype='D') # dy(Qzz) vanishes 
	LHS[8*M-1] = np.zeros(8*M, dtype='D')
	LHS[7*M,RQzz] = D1[0]
	LHS[8*M-1,RQzz] = D1[M-1]

	RHS[0]     = np.zeros(8*M,dtype='D')
	RHS[M-1]   = np.zeros(8*M,dtype='D')
	RHS[M]     = np.zeros(8*M,dtype='D')
	RHS[2*M-1] = np.zeros(8*M,dtype='D')
	RHS[2*M]   = np.zeros(8*M,dtype='D')
	RHS[3*M-1] = np.zeros(8*M,dtype='D')
	RHS[3*M]   = np.zeros(8*M,dtype='D')
	RHS[4*M-1] = np.zeros(8*M,dtype='D')
	RHS[4*M]   = np.zeros(8*M,dtype='D')
	RHS[5*M-1] = np.zeros(8*M,dtype='D')
	RHS[5*M]   = np.zeros(8*M,dtype='D')
	RHS[6*M-1] = np.zeros(8*M,dtype='D')
	RHS[6*M]   = np.zeros(8*M,dtype='D')
	RHS[7*M-1] = np.zeros(8*M,dtype='D')
	RHS[7*M]   = np.zeros(8*M,dtype='D')
	RHS[8*M-1] = np.zeros(8*M,dtype='D')
	
	_spec = eig(LHS,RHS,left=0,right=1)

	_eig_list = _spec[0]
	finite_idx = np.where(np.isfinite(_eig_list))[0]
	_modes_list = _spec[1]

	# clean the eigenvalue list of all infinities
	clean_eig_list = _eig_list[finite_idx]
	print("There are",len(finite_idx),"finite eigenvalues")
	print("The shape of the eigenmodes is",_modes_list.shape)
	clean_modes_list = _modes_list[:,finite_idx]
	print("The shape of the cleaned eigenmodes is",clean_modes_list.shape)
	# clean_eig_list = list(filter(lambda ev: np.isfinite(ev), _eig_list))
	
	return (clean_eig_list, clean_modes_list)

#print("at k={}, gammadot={}, tau={}, tau_a={}, fastest growth rate:{:.4f} @ freq={:.4f}"\
#.format(k,_gammadot,_tau,_tau_a,np.real(max_val),np.imag(max_val)))
#print("at k={}, tBar={}, aBar={}, fastest growth rate:{:.4f} @ freq={:.4f}"\
#.format(k,_gammadot*_tau,1/(_gammadot*_tau_a),np.real(max_val),np.imag(max_val)))

"""
f=open('spectrum.txt','w')
for i in range(len(_eig_list)):
	if np.isfinite(_eig_list[i]):
	f.write('%20.18f %20.18f\n'%(np.real(_eig_list[i]),np.imag(_eig_list[i]))    )
f.close()

f=open('list.txt','w')
for i in range(len(_eig_list)):
	f.write('%d %20.18f %20.18f\n'%(i,np.real(_eig_list[i]),np.imag(_eig_list[i]))    )    
f.close()
  """


## OUTPUT
"""
_my_mode=_modes_list[:,mode_number]

_psi=_my_mode[0:M]
_Qxx=_my_mode[M:2*M]
_Qxy=_my_mode[2*M:3*M]

if do_plot_mode:

	f=open('psi.field','w')
	for m in range(M):
	f.write('%f %20.18f %20.18f\n'%(ygl[m],np.real(_psi[m]),np.imag(_psi[m])))
	f.close()

	f=open('Qxx.field','w')
	for m in range(M):
	f.write('%f %20.18f %20.18f\n'%(ygl[m],np.real(_Qxx[m]),np.imag(_Qxx[m])))
	f.close()

	f=open('Qxy.field','w')
	for m in range(M):
	f.write('%f %20.18f %20.18f\n'%(ygl[m],np.real(_Qxy[m]),np.imag(_Qxy[m])))
	f.close()
"""
