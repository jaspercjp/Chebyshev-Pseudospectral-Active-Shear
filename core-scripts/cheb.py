import numpy as np

def cheb(M):
    cbar = np.ones(M,dtype='d')
    cbar[0] = 2.0
    cbar[M-1] = 2.0

	# Chebyshev grid points
    ygl = np.zeros(M,dtype='d')
    for m in range(M):
        ygl[m] = np.cos(np.pi*m/(M-1))

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

    # The factor 2 takes care of the domain being from -1/2 to 1/2
    return (D1, ygl)