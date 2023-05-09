import numpy as np
from scipy.linalg import eig
import numpy as np
from cheb import cheb
import matplotlib.pyplot as plt

##------------------------------------
def spectrum(kx,a,_ell_over_W_squared=0.01, M=50):
	"""Takes in a set of parameters and returns the spectrum that 
	corresponds to these parameter values. The spectrum is calculated at 
	zero shear in a three dimensional system, with y being the wall-normal
	coordinate

	Args:
		kx (float): perturbation wavenumber
		a (float): activity

	Returns:
		list[complx numbers]: a (cleaned of any infinities) list of eigenvalues
	"""
	# M is the resolution

	# parameters
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
	# print("There are",len(finite_idx),"finite eigenvalues")
	# print("The shape of the eigenmodes is",_modes_list.shape)
	clean_modes_list = _modes_list[:,finite_idx]
	# print("The shape of the cleaned eigenmodes is",clean_modes_list.shape)
	# clean_eig_list = list(filter(lambda ev: np.isfinite(ev), _eig_list))
	
	return (clean_eig_list, clean_modes_list)

def plot_modes(idx, ygl, evecs, M):
    """ Plots the mode at index "idx" on the coordinates "ygl" """
    plt.figure()
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

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()