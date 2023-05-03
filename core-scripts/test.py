from zeroShearSpectrum3D import spectrum
import numpy as np
import matplotlib.pyplot as plt
from cheb import cheb

evals, evecs = spectrum(0.4, 5)
# print(evals)
max_rate = np.max(np.real(evals))
idx = np.where(np.real(evals)==max_rate)[0]
print(max_rate)
M = 50
D1, ygl = cheb(M)
plt.figure()
fig, axs = plt.subplots(3, 3)
Vx = np.reshape(evecs[idx,0:M], -1)
axs[0,0].plot(ygl, np.real(Vx))
axs[0,0].plot(ygl, np.imag(Vx))

Vy = np.reshape(evecs[idx,M:2*M], -1)
axs[0,1].plot(ygl, np.real(Vy))
axs[0,1].plot(ygl, np.imag(Vy))
axs[0,1].title.set_text('$V_y$')

Vz = np.reshape(evecs[idx,2*M:3*M], -1)
axs[0,2].plot(ygl, np.real(Vz))
axs[0,2].plot(ygl, np.imag(Vz))

Qxx = np.reshape(evecs[idx,3*M:4*M], -1)
axs[1,0].plot(ygl, np.real(Qxx))
axs[1,0].plot(ygl, np.imag(Qxx))

Qxy = np.reshape(evecs[idx,4*M:5*M], -1)
axs[1,1].plot(ygl, np.real(Qxy))
axs[1,1].plot(ygl, np.imag(Qxy))

Qxz = np.reshape(evecs[idx,5*M:6*M], -1)
axs[1,2].plot(ygl, np.real(Qxz))
axs[1,2].plot(ygl, np.imag(Qxz))

Qyz = np.reshape(evecs[idx,6*M:7*M], -1)
axs[2,0].plot(ygl, np.real(Qyz))
axs[2,0].plot(ygl, np.imag(Qyz))

Qzz = np.reshape(evecs[idx,7*M:8*M], -1)
axs[2,1].plot(ygl, np.real(Qzz))
axs[2,1].plot(ygl, np.imag(Qzz))
plt.savefig("test-eigenmodes.jpg")