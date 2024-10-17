import astropy.io.fits as asp
import matplotlib.pyplot as plt
import scipy.integrate as spInt
import numpy as np
from datetime import datetime


hdu_list = asp.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data # same for every galaxy
flux = hdu_list['FLUX'].data # different per galaxy

for i in range(5):
    plt.plot(logwave, flux[i])
    
plt.xlabel(r'log($\lambda$)')
plt.ylabel('Spectrum')
plt.title('Spectrum of galaxies from dataset vs log of wavelength')
plt.show()


# part (b)

new_flux = []
f_areaList = []
f_meanList = []

N_gal = len(flux)
N_wave = len(logwave)

for i in range(N_gal):
    f_area = spInt.trapezoid(flux[i], logwave)
    f_areaList.append(f_area)
    new_flux.append(np.array(flux[i] / f_area))


new_flux = np.array(new_flux)


# part (c)

for i in range(N_wave):
    f_mean = new_flux[:, i].mean()
    f_meanList.append(f_mean)
    new_flux[:, i] -= f_mean


# part (d)

R = new_flux.T
print(R.shape)
C = (R @ R.T)

startTime = datetime.now()
evals, evecs = np.linalg.eigh(C) # evals are in order from largest to smallest, evecs[:, i] corresponds to evals[i]

# Now reorder them in descending order
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

print(f"By direct diagonalization of C, eigenvectors are found in {(datetime.now()-startTime)}")

plt.imshow(evecs[:, 0:5], aspect=0.00125)
plt.colorbar()
plt.xlabel('Eigenvector')
plt.ylabel('Value')
plt.title('First five eigenvectors of C')
plt.show()


# part (e)
startTime = datetime.now()
(U, W, VT) = np.linalg.svd(R.T) # we use R.T because of the new definition of C given in part (e), and here VT is equivalent to evecs
print(f"By SVD on R.T, eigenvectors are found in {(datetime.now()-startTime)}")

print(U.shape)
print(W.shape)
print(VT.shape)

V = VT.T

idx = np.argsort(np.square(W))[::-1]
W = W[idx]
V = V[:, idx]

VT = V.T

print(R)
# print(V - evecs)
print(V)
print(VT)
print(evecs)

"""  
Looking at the eigenvalues individually, seems like they're equivalent across methods, up to a factor of -1. Need to figure out
"""


# part (f)

"""  
Need to think about. No coding required
"""


# part (g)

"""  
As stated in the problem, to get the list of coefficients, c_i, we just rotate evals into the eigenspectrum basis
"""

N_c = 5

""" cList = VT.dot(R.T)
currCList = cList[0:N_c] """

Rinv = V @ np.diag(1. / W) @ U.T

evalSVD = np.square(W)
cList = V.T @ R.T

""" print(cList)
print(R.T.dot(VT.T)) """

galIdx = 0


print(cList[galIdx])


print(f_meanList + ((cList @ evalSVD) * f_areaList[galIdx]))
# print((f_meanList + (cList) * f_areaList[galIdx])
print(flux[galIdx])
print(new_flux[galIdx])

