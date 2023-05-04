from zeroShearSpectrum3D import spectrum
import numpy as np
import matplotlib.pyplot as plt
from cheb import cheb
k,a = 1.25, 1.09
evals_low, evecs_low = spectrum(k,a,M=50)
evals_high, evecs_high = spectrum(k,a,M=100)
# print(len(evals_low), len(evals_high))
re_evals_low = np.real(evals_low)
re_evals_high = np.real(evals_high)
re_evals_low = re_evals_low[np.argsort(-re_evals_low)]
re_evals_high  = re_evals_high[np.argsort(-re_evals_high)]

# n = min(len(re_evals_low), len(re_evals_high))
# diffs = re_evals_low[0:n] - re_evals_high[0:n]
# print(diffs)
# plt.plot()

# for ev in evals:
#     print(ev)
# max_rate = np.max(np.real(evals))
# idx = np.where(np.real(evals)==max_rate)[0]
# print("The maximum growth rate at (kx,a)=({},{}) is {}".format(k,a,max_rate))

# M = 50
# D1, ygl = cheb(M)

## Plot a mode stored in evecs
# evecs = evecs_high
# plt.figure()
# fig, axs = plt.subplots(3, 3)
# Vx = np.reshape(evecs[idx,0:M], -1)
# axs[0,0].plot(ygl, np.real(Vx))
# axs[0,0].plot(ygl, np.imag(Vx))

# Vy = np.reshape(evecs[idx,M:2*M], -1)
# axs[0,1].plot(ygl, np.real(Vy))
# axs[0,1].plot(ygl, np.imag(Vy))
# axs[0,1].title.set_text('$V_y$')

# Vz = np.reshape(evecs[idx,2*M:3*M], -1)
# axs[0,2].plot(ygl, np.real(Vz))
# axs[0,2].plot(ygl, np.imag(Vz))

# Qxx = np.reshape(evecs[idx,3*M:4*M], -1)
# axs[1,0].plot(ygl, np.real(Qxx))
# axs[1,0].plot(ygl, np.imag(Qxx))

# Qxy = np.reshape(evecs[idx,4*M:5*M], -1)
# axs[1,1].plot(ygl, np.real(Qxy))
# axs[1,1].plot(ygl, np.imag(Qxy))

# Qxz = np.reshape(evecs[idx,5*M:6*M], -1)
# axs[1,2].plot(ygl, np.real(Qxz))
# axs[1,2].plot(ygl, np.imag(Qxz))

# Qyz = np.reshape(evecs[idx,6*M:7*M], -1)
# axs[2,0].plot(ygl, np.real(Qyz))
# axs[2,0].plot(ygl, np.imag(Qyz))

# Qzz = np.reshape(evecs[idx,7*M:8*M], -1)
# axs[2,1].plot(ygl, np.real(Qzz))
# axs[2,1].plot(ygl, np.imag(Qzz))
# plt.savefig("test-eigenmodes.jpg")