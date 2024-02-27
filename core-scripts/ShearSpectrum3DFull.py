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
	ksq = kx*kx + kz*kz
	II_aux = -(3+2*gd*gd)*(1 + _ell_over_W_squared*ksq + 1j*kx*gd*ygl)*II
	D2_aux =  _ell_over_W_squared*(3+2*gd*gd)*D2
	SPEC_OP_2 = II_aux + D2_aux

	# Variable layout
	RV1  = slice(0*M,1*M)
	RV2 = slice(1*M, 2*M)
	RP = slice(2*M, 3*M)
	RQxx  = slice(3*M,4*M)
	RQxy  = slice(4*M,5*M)
	RQxz = slice(5*M, 6*M)
	RQyz = slice(6*M, 7*M)
	RQzz = slice(7*M, 8*M)

	LHS = np.zeros((8*M,8*M),dtype=np.complex128)

	#V1 eqn
	LHS[RV1, RV1] = - ksq*II*eta + eta*D2
	LHS[RV1, RP] = 1j*kx*eta*II
	LHS[RV1, RQxx] = - 1j*a*kx*II
	LHS[RV1, RQxy] = - a*D1
	LHS[RV1, RQxz] = - 1j*a*kz*II

	#V2 eqn
	LHS[RV2, RV2] = - ksq*II*eta + eta*D2
	LHS[RV2, RP] = eta*D1
	LHS[RV2, RQxx] = a*D1
	LHS[RV2, RQxy] = - 1j*a*kx*II
	LHS[RV2, RQyz] = - 1j*a*kz*II
	LHS[RV2, RQzz] = a*D1

	#P eqn
	LHS[RP, RV1] = kx*ksq*II*eta - kx*eta*D2
	LHS[RP, RV2] = 1j*eta*(D3+kz**2*II) - 1j*ksq*eta*D1
	LHS[RP, RP] = 1j*eta*(D3+kz**2*II)
	LHS[RP, RQxz] = - 1j*a*kx*kz*II
	LHS[RP, RQyz] = - a*kz*D1
	LHS[RP, RQzz] = - 1j*a*kz**2*II

	#Qxx eqn
	LHS[RQxx, RV1] = 2j*kx*II*(3+4*gd**2) + 4*gd*D1
	LHS[RQxx, RV2] = - 2j*kx*II*gd
	LHS[RQxx, RQxx] = SPEC_OP_2
	LHS[RQxx, RQxy] = - 4/3*II*gd*(-3-2*gd**2)

	#Qxy eqn
	LHS[RQxy, RV1] = 3j*kx*II*gd + 3*D1
	LHS[RQxy, RV2] = 3j*kx*II*(1+2*gd**2) + 3*gd*D1
	LHS[RQxy, RQxx] = II*gd*(-3-2*gd**2)
	LHS[RQxy, RQxy] = SPEC_OP_2
	LHS[RQxy, RQzz] = II*gd*(-3-2*gd**2)

	#Qxz eqn
	LHS[RQxz, RV1] = 3j*II*(2*kz**2 - kx**2*(2+4*gd**2))/(2*kz) - (3*kx*gd*D1)/kz
	LHS[RQxz, RV2] = - (3*kx*(2+4*gd**2)*D1)/(2*kz) + (3j*gd*D2)/kz
	LHS[RQxz, RQxz] = SPEC_OP_2
	LHS[RQxz, RQyz] = - II*gd*(-3-2*gd**2)

	#Qyz eqn
	LHS[RQyz, RV1] = - (3j*kx**2*II*gd)/kz - (3*kx*D1)/kz
	LHS[RQyz, RV2] = 3j*kz*II - (3*kx*gd*D1)/kz + (3j*D2)/kz
	LHS[RQyz, RQyz] = SPEC_OP_2

	#Qzz eqn
	LHS[RQzz, RV1] = -2j*kx*II*(3+2*gd**2) - 2*gd*D1
	LHS[RQzz, RV2] = -2j*kx*II*gd - 6*D1
	LHS[RQzz, RQxy] = 2/3*II*gd*(-3-2*gd**2)
	LHS[RQzz, RQzz] = SPEC_OP_2

	if False:
		print("V1 Op:", np.linalg.cond(LHS[RV1,:]))
		print("V2 Op:", np.linalg.cond(LHS[RV2,:]))
		print("P Op:", np.linalg.cond(LHS[RP,:]))

		print("Qxx Op:", np.linalg.cond(LHS[RQxx,:]))
		print("Qxy Op:", np.linalg.cond(LHS[RQxy,:]))
		print("Qxz Op:", np.linalg.cond(LHS[RQxz,:]))
		print("Qyz Op:", np.linalg.cond(LHS[RQyz,:]))
		print("Qzz Op:", np.linalg.cond(LHS[RQzz,:]))

	RHS = np.zeros((8*M,8*M),dtype='D')
	RHS[RQxx,RQxx] = (3+2*gd*gd)*II
	RHS[RQxy,RQxy] = (3+2*gd*gd)*II 
	RHS[RQxz,RQxz] = (3+2*gd*gd)*II
	RHS[RQyz,RQyz] = (3+2*gd*gd)*II
	RHS[RQzz,RQzz] = (3+2*gd*gd)*II

	## =========== Boundary conditions ===============

	# V1 vanishes at boundary
	LHS[0]     = np.zeros(8*M,dtype='D')
	LHS[M-1]   = np.zeros(8*M,dtype='D')
	LHS[0,0]      = 1.0
	LHS[M-1,M-1]  = 1.0
	RHS[0]     = np.zeros(8*M,dtype='D')
	RHS[M-1]   = np.zeros(8*M,dtype='D')

	# V2 vanishes at boundary
	LHS[M]     = np.zeros(8*M,dtype='D') 
	LHS[2*M-1] = np.zeros(8*M,dtype='D')
	LHS[M,M]     = 1.0
	LHS[2*M-1,2*M-1] = 1.0
	RHS[M]     = np.zeros(8*M,dtype='D')
	RHS[2*M-1] = np.zeros(8*M,dtype='D')

	# V3 vanishes at the boundary
	# This condition is expressed through V2 prime by the incompresibility condition
	LHS[2*M, RV2]   = D1[0,:]
	# LHS[3*M-1, RV2] = D1[M-1,:]
	RHS[2*M]   = np.zeros(8*M,dtype='D')
	# RHS[3*M-1] = np.zeros(8*M,dtype='D')

	# dy(Qxx) vanishes at the boundary
	LHS[3*M] = np.zeros(8*M,dtype='D') 
	LHS[4*M-1] = np.zeros(8*M,dtype='D')
	LHS[3*M, RQxx] = D1[0]
	LHS[4*M-1, RQxx] = D1[M-1]
	RHS[3*M]   = np.zeros(8*M,dtype='D')
	RHS[4*M-1] = np.zeros(8*M,dtype='D')

	# dy(Qxy) vanishes at the boundary
	LHS[4*M] = np.zeros(8*M, dtype='D')
	LHS[5*M-1] = np.zeros(8*M, dtype='D')
	LHS[4*M, RQxy] = D1[0]
	LHS[5*M-1, RQxy] = D1[M-1]
	RHS[4*M]   = np.zeros(8*M,dtype='D')
	RHS[5*M-1] = np.zeros(8*M,dtype='D')

	# dy(Qxz) vanishes at the boundary
	LHS[5*M] = np.zeros(8*M, dtype='D') 
	LHS[6*M-1] = np.zeros(8*M,dtype='D')
	LHS[5*M,RQxz] = D1[0]
	LHS[6*M-1,RQxz] = D1[M-1]
	RHS[5*M]   = np.zeros(8*M,dtype='D')
	RHS[6*M-1] = np.zeros(8*M,dtype='D')

	# dy(Qyz) vanishes at the boundary
	LHS[6*M] = np.zeros(8*M,dtype='D') 
	LHS[7*M-1] = np.zeros(8*M,dtype='D')
	LHS[6*M,RQyz] = D1[0]
	LHS[7*M-1,RQyz] = D1[M-1]
	RHS[6*M]   = np.zeros(8*M,dtype='D')
	RHS[7*M-1] = np.zeros(8*M,dtype='D')

	# dy(Qzz) vanishes at the boundary
	LHS[7*M] = np.zeros(8*M, dtype='D')
	LHS[8*M-1] = np.zeros(8*M, dtype='D')
	LHS[7*M,RQzz] = D1[0]
	LHS[8*M-1,RQzz] = D1[M-1]
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