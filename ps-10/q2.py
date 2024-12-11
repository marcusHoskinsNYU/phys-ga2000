import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import scipy as sp
from dcst import dst, idst

"""  
Part (a)
"""


def initialPsi(x, L, t0 = 0):
    # initial form of wavefunction, x here is a np array
    x0 = L/2.
    sigma = 1e-10
    kappa = 5e10
    psi0 = np.exp(-((x - x0)**2) / (2 * sigma**2)) * np.exp(1j * kappa * x)

    return psi0

def xPosList(a, N):
    # returns the position value of the ith spatial slice of our lattice
    # i ranges from 0 to N-1
    xList = [i * a for i in range(N)]
    return xList


def spectralCoeff(psi0):
    # computes the spectral-method time evolution of the given initial wavefunction and returns a list of
    psi0Real = np.real(psi0)
    psi0Imag = np.imag(psi0)

    alphaCoeffList = dst(psi0Real)
    etaCoeffList = dst(psi0Imag)

    return alphaCoeffList, etaCoeffList

def energy(k, L, hbar, m):
    return np.pi**2 * hbar * k**2 / (2 * m * L**2)

def realPsiAtT(t, n, psi0, L, hbar, m, N):
    # returns the real part of psi(x_n, t)
    alphaCoeffList, etaCoeffList = spectralCoeff(psi0)

    realPsi = 0

    for k in range(1, N):
        realPsi += ((alphaCoeffList[k] * np.cos(energy(k, L, hbar, m)*t)) + (etaCoeffList[k] * np.sin(energy(k, L, hbar, m)*t))) * np.sin(np.pi * k * n / N)

    return realPsi / N

def realPsiAtTviaTransform(t, psi0, L, hbar, m, N):
    # returns the real part of psi(x_n, t) using idst
    alphaCoeffList, etaCoeffList = spectralCoeff(psi0)

    coeffList = [0]

    for k in range(1, N):
        coeffList.append((alphaCoeffList[k] * np.cos(energy(k, L, hbar, m)*t)) + (etaCoeffList[k] * np.sin(energy(k, L, hbar, m)*t)))

    coeffList = np.array(coeffList, dtype = complex)

    realPsi = idst(coeffList)

    return realPsi

# def ifftRealPsiAtT()

"""  
Part (b)
"""

# || CONSTANTS
M = 9.109e-31       # mass of electron in kg
L = 1e-8            # length of box in m
t0 = 0              # initial time 
N = 1000            # number of spatial slices
a = L / N           # spacing of aptial grid points
h = 1e-18           # time step in s
hbar = 1.054e-34    # hbar
scale = 1e-9        # scale of the wavefunction values so they appear reasonable on the screen

x = np.linspace(0, L, N)
psi0 = initialPsi(x, L)
t = 1e-16
psiT = []

for n in range(N):
    psiT.append(realPsiAtT(t, n, psi0, L, hbar, M, N))

psiT_idst = realPsiAtTviaTransform(t, psi0, L, hbar, M, N)

plt.plot(x/scale, psiT, label='From Formula', linewidth=2)
plt.plot(x/scale, psiT_idst, '--', label='From IDST', linewidth=2)
plt.xlabel('Position, x [nm]', size=20)
plt.ylabel(r'Re($\psi (x, t)$)', size=20)
plt.title(rf'Re($\psi (x, t)$) vs Position, at t = {t:.2E} s', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(prop={'size':20})
plt.show()



"""  
Part (c)
"""
steps = 3000        # number of time steps taken
nframes = 10

psiList = []

for i in range(steps):
    psiList.append(realPsiAtTviaTransform(i*h, psi0, L, hbar, M, N))

newSteps = int(steps / nframes)

plt.plot(x/scale, np.real(psiList[-1]), linewidth=2)
plt.xlabel('Position, x [nm]', size=20)
plt.ylabel(r'Re($\psi (x, t)$)', size=20)
plt.title(rf'Re($\psi (x, t)$) vs Position, at t = {h*((10)*newSteps):.2E} s, Spectral', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

for i in range(nframes+1):
    plt.plot(x/scale, np.real(psiList[(i)*newSteps]), linewidth=2)
    plt.xlabel('Position, x [nm]', size=20)
    plt.ylabel(r'Re($\psi (x, t)$)', size=20)
    plt.title(rf'Re($\psi (x, t)$) vs Position, at t = {h*((i)*newSteps):.2E} s, Spectral', size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()


 
"""  
Part (d)
"""

steps = 100000

plt.plot(x/scale, np.real(realPsiAtTviaTransform(steps*h, psi0, L, hbar, M, N)), linewidth=2)
plt.xlabel('Position, x [nm]', size=20)
plt.ylabel(r'Re($\psi (x, t)$)', size=20)
plt.title(rf'Re($\psi (x, t)$) vs Position, at t = {h*steps:.2E} s, Spectral', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()