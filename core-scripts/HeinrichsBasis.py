from cheb import cheb
import numpy as np

class HeinrichsBasis:
    def __init__(self, M):
        II = np.identity(M,dtype='d')
        self.II = II

        D1, y = cheb(M)
        self.y = y
        S = np.zeros(M)
        S[1:M-1] = 1 / (1 - y[1:M-1]**2)**2
        D1[0,:]=0; D1[:,0]=0; 
        D1[M-1,:]=0; D1[:,M-1]=0
        
        D2 = D1@D1
        D2[0,:]=0; D2[:,0]=0
        D2[M-1,:]=0; D2[:,M-1]=0

        D3 = D2@D1
        D3[0,:]=0; D3[:,0]=0
        D3[M-1,:]=0; D3[:,M-1]=0
        
        D4 = D2@D2
        D4[0,:]=0; D4[:,0]=0; 
        D4[M-1,:]=0; D4[:,M-1]=0

        # Calculate D1 matrix in new basis
        D1_H0 = 4*y*(-1+y**2)*II
        D1_H1 = ((-1+y**2)**2 * D1.T).T
        self.D1 = (D1_H0 + D1_H1) * S

        # D2 Matrix 
        D2_H0 = 4*(-1+3*y**2)*II
        D2_H1 = (8*y*(-1+y**2) * D1.T).T
        D2_H2 = ((-1+y**2)**2 * D2.T).T
        self.D2 = (D2_H0 + D2_H1 + D2_H2) * S

        # D3 Matrix 
        D3_H0 = 24*y*II 
        D3_H1 = (12 * (-1+3*y**2) * D1.T).T
        D3_H2 = (12*y*(-1+y**2) * D2.T).T
        D3_H3 = ((-1+y**2)**2 * D3.T).T
        self.D3 = (D3_H0 + D3_H1 + D3_H2 + D3_H3) * S

        # D4 Matrix
        D4_H0 = 24*II 
        D4_H1 = (96*y * D1.T).T
        D4_H2 = (24*(-1+3*y**2) * D2.T).T
        D4_H3 = (16*y*(-1+y**2) * D3.T).T
        D4_H4 = ((-1+y**2)**2 * D4.T).T
        self.D4 = (D4_H0 + D4_H1 + D4_H2 + D4_H3 + D4_H4) * S