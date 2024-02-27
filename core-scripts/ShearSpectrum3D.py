import numpy as np
from scipy.linalg import eig
import numpy as np
from cheb import cheb
import matplotlib.pyplot as plt
from decimal import Decimal

def spectrum(kx, kz, gd, a, _ell_over_W_squared=0.01, M=50):
	"""
		Takes in a set of parameters and returns the spectrum that 
		corresponds to these parameter values.
		
		NOTE: These equations assume tau=1 and lambda=1!!!
	"""
	# parameters
	eta = 1.0
	##------------------------------------

	II = np.identity(M)

	# Build Chebyshev differentiation matrix and the grid points on [-1, 1]
	D1, ygl = cheb(M)
	D2 = D1@D1
	D3 = D2@D1
	D4 = D3@D1

	# Some convenient definitions
	II_aux = -(3+2*gd*gd)*(1 + _ell_over_W_squared*(kx*kx+kz*kz) + 1j*kx*gd*ygl)*II
	D2_aux =  _ell_over_W_squared*(3+2*gd*gd)*D2
	SPEC_OP_2 = II_aux + D2_aux
	ksq = kx*kx + kz*kz

	# Variable layout
	RVx  = slice(0*M,1*M)
	RVy = slice(1*M, 2*M)
	RVz = slice(2*M, 3*M)
	RQxx  = slice(3*M,4*M)
	RQxy  = slice(4*M,5*M)
	RQxz = slice(5*M, 6*M)
	RQyz = slice(6*M, 7*M)
	RQzz = slice(7*M, 8*M)

	LHS = np.zeros((8*M,8*M),dtype=np.complex128)

	#V1 eqn
	LHS[RVx, RVx]= kx*D3*eta - ksq*kx*eta*D1
	LHS[RVx, RVy]= - 1j*ksq*kz**2*II*eta - 1j*D4*eta + 1j*(kx**2+2*kz**2)*eta*D2
	LHS[RVx, RQxx]= 1j*a*kz**2*D1
	LHS[RVx, RQxy]= a*kx*kz**2*II
	LHS[RVx, RQxz]= 1j*a*kx*kz*D1
	LHS[RVx, RQyz]= a*kz**3*II + a*kz*D2
	LHS[RVx, RQzz]= 2*1j*a*kz**2*D1

	#V2 eqn
	LHS[RVy, RVx]= 1j*ksq*ksq*II*eta - 1j*ksq*eta*D2
	LHS[RVy, RVy]= - kx*D3*eta + ksq*kx*eta*D1
	LHS[RVy, RQxx]= - a*kx*kz**2*II
	LHS[RVy, RQxy]= 1j*a*kz**2*D1
	LHS[RVy, RQxz]= a*kz*(kx**2-kz**2)*II
	LHS[RVy, RQyz]= - 1j*a*kx*kz*D1
	LHS[RVy, RQzz]= a*kx*kz**2*II

	#V3 eqn
	LHS[RVz, RVx]= kz*D3*eta - ksq*kz*eta*D1
	LHS[RVz, RVy]= 1j*ksq*kx*kz*II*eta - 1j*kx*kz*eta*D2
	LHS[RVz, RQxx]= - 2*1j*a*kx*kz*D1
	LHS[RVz, RQxy] = - a*kx**2*kz*II - a*kz*D2
	LHS[RVz, RQxz] = - 1j*a*kz**2*D1
	LHS[RVz, RQyz] = - a*kx*kz**2*II
	LHS[RVz, RQzz] = - 1j*a*kx*kz*D1

	#Qxx eqn
	LHS[RQxx, RVx] = 2*1j*kx*II*(3+4*gd**2) + 4*gd*D1
	LHS[RQxx, RVy] = - 2*1j*kx*II*gd
	LHS[RQxx, RQxx] = SPEC_OP_2
	LHS[RQxx, RQxy] = - 4/3*II*gd*(-3-2*gd**2)

	#Qxy eqn
	LHS[RQxy, RVx] = 3*1j*kx*II*gd + 3*D1
	LHS[RQxy, RVy] = 3*1j*kx*II*(1+2*gd**2) + 3*gd*D1
	LHS[RQxy, RQxx] = II*gd*(-3-2*gd**2)
	LHS[RQxy, RQxy] = SPEC_OP_2
	LHS[RQxy, RQzz] = II*gd*(-3-2*gd**2)

	#Qxz eqn
	LHS[RQxz, RVx] = 3/2*1j*II*(2*kz**2 - 2*kx*kx*(1+2*gd**2)) - 3*kx*gd*D1
	LHS[RQxz, RVy] = - 3/2*kx*(2+4*gd**2)*D1 + 3*1j*gd*D2
	LHS[RQxz, RQxz] = kz*SPEC_OP_2
	LHS[RQxz, RQyz] = - kz*II*gd*(-3-2*gd**2)

	#Qyz eqn
	LHS[RQyz, RVx]= - 3*1j*kx**2*II*gd - 3*kx*D1
	LHS[RQyz, RVy]= 3*1j*kz**2*II - 3*kx*gd*D1 + 3*1j*D2
	LHS[RQyz, RQyz] = kz*SPEC_OP_2

	#Qzz eqn
	LHS[RQzz, RVx]= - 2*1j*kx*kz*II*(3+2*gd**2) - 2*kz*gd*D1
	LHS[RQzz, RVy]= - 2*1j*kx*kz*II*gd - 6*kz*D1
	LHS[RQzz, RQxy]= 2/3*kz*II*gd*(-3-2*gd**2)
	LHS[RQzz, RQzz] = kz*SPEC_OP_2

	if False:
		print("Vx Op:", np.linalg.cond(LHS[RVx,:]))
		print("Vy Op:", np.linalg.cond(LHS[RVy,:]))
		print("Vz Op:", np.linalg.cond(LHS[RVz,:]))

		print("Qxx Op:", np.linalg.cond(LHS[RQxx,:]))
		print("Qxy Op:", np.linalg.cond(LHS[RQxy,:]))
		print("Qxz Op:", np.linalg.cond(LHS[RQxz,:]))
		print("Qyz Op:", np.linalg.cond(LHS[RQyz,:]))
		print("Qzz Op:", np.linalg.cond(LHS[RQzz,:]))

	RHS = np.zeros((8*M,8*M),dtype='D')
	RHS[RQxx,RQxx] = (3+2*gd*gd)*II
	RHS[RQxy,RQxy] = (3+2*gd*gd)*II 
	RHS[RQxz,RQxz] = kz*(3+2*gd*gd)*II
	RHS[RQyz,RQyz] = kz*(3+2*gd*gd)*II
	RHS[RQzz,RQzz] = kz*(3+2*gd*gd)*II

	## =========== Boundary conditions ===============

	# Vx vanishes at boundary
	LHS[0]     = np.zeros(8*M,dtype='D')
	LHS[M-1]   = np.zeros(8*M,dtype='D')
	LHS[0,0]      = 1.0
	LHS[M-1,M-1]  = 1.0
	RHS[0]     = np.zeros(8*M,dtype='D')
	RHS[M-1]   = np.zeros(8*M,dtype='D')
	
	# NOTE: Since equation is third order in vx and vz, also impose a Neumann condition
	# LHS[1,RVx] = D1[0,:]
	# RHS[1,:] = np.zeros(8*M)

	# Vy vanishes at boundary
	LHS[M]     = np.zeros(8*M,dtype='D') 
	LHS[2*M-1] = np.zeros(8*M,dtype='D')
	LHS[M,M]     = 1.0
	LHS[2*M-1,2*M-1] = 1.0
	RHS[M]     = np.zeros(8*M,dtype='D')
	RHS[2*M-1] = np.zeros(8*M,dtype='D')

	# Vz vanishes at the boundary
	# LHS[2*M]   = np.zeros(8*M,dtype='D')
	# LHS[3*M-1] = np.zeros(8*M,dtype='D')
	# LHS[2*M,2*M] = 1.0
	# LHS[3*M-1,3*M-1] = 1.0
	# RHS[2*M]   = np.zeros(8*M,dtype='D')
	# RHS[3*M-1] = np.zeros(8*M,dtype='D')

	# NOTE: Since equation is third order in vx and vz, also impose a Neumann condition
	# LHS[2*M+1,RVz] = D1[0,:]
	# RHS[2*M+1,:] = np.zeros(8*M)

	# dy(Qxx) vanishes at the boundary
	LHS[3*M] = np.zeros(8*M,dtype='D') 
	LHS[4*M-1] = np.zeros(8*M,dtype='D')
	LHS[3*M, RQxx] = D1[0]
	LHS[4*M-1, RQxx] = D1[M-1]
	# LHS[3*M, 3*M] = 1.0
	# LHS[4*M-1, 4*M-1] = 1.0
	RHS[3*M]   = np.zeros(8*M,dtype='D')
	RHS[4*M-1] = np.zeros(8*M,dtype='D')

	# dy(Qxy) vanishes at the boundary
	LHS[4*M] = np.zeros(8*M, dtype='D')
	LHS[5*M-1] = np.zeros(8*M, dtype='D')
	LHS[4*M, RQxy] = D1[0]
	LHS[5*M-1, RQxy] = D1[M-1]
	# LHS[4*M, 4*M] = 1.0
	# LHS[5*M-1, 5*M-1] = 1.0
	RHS[4*M]   = np.zeros(8*M,dtype='D')
	RHS[5*M-1] = np.zeros(8*M,dtype='D')

	# dy(Qxz) vanishes at the boundary
	LHS[5*M] = np.zeros(8*M, dtype='D') 
	LHS[6*M-1] = np.zeros(8*M,dtype='D')
	LHS[5*M,RQxz] = D1[0]
	LHS[6*M-1,RQxz] = D1[M-1]
	# LHS[5*M, 5*M] = 1.0
	# LHS[6*M-1, 6*M-1] = 1.0
	RHS[5*M]   = np.zeros(8*M,dtype='D')
	RHS[6*M-1] = np.zeros(8*M,dtype='D')

	# dy(Qyz) vanishes at the boundary
	LHS[6*M] = np.zeros(8*M,dtype='D') 
	LHS[7*M-1] = np.zeros(8*M,dtype='D')
	LHS[6*M,RQyz] = D1[0]
	LHS[7*M-1,RQyz] = D1[M-1]
	# LHS[6*M, 6*M] = 1.0
	# LHS[7*M-1, 7*M-1] = 1.0
	RHS[6*M]   = np.zeros(8*M,dtype='D')
	RHS[7*M-1] = np.zeros(8*M,dtype='D')

	# dy(Qzz) vanishes at the boundary
	LHS[7*M] = np.zeros(8*M, dtype='D')
	LHS[8*M-1] = np.zeros(8*M, dtype='D')
	LHS[7*M,RQzz] = D1[0]
	LHS[8*M-1,RQzz] = D1[M-1]
	# LHS[7*M, 7*M] = 1.0
	# LHS[8*M-1, 8*M-1] = 1.0
	RHS[7*M]   = np.zeros(8*M,dtype='D')
	RHS[8*M-1] = np.zeros(8*M,dtype='D')

	# print("Spectral Operator Condition Number:%.2E" % Decimal(np.linalg.cond(LHS)))
	_spec = eig(LHS,RHS,left=0,right=1)

	_eig_list = _spec[0]
	finite_idx = np.where(np.isfinite(_eig_list))[0]
	_modes_list = _spec[1]

	# clean the eigenvalue list of all infinities
	clean_eig_list = _eig_list[finite_idx]
	clean_modes_list = _modes_list[:,finite_idx]

	return (clean_eig_list, clean_modes_list)