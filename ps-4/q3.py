import math as m
import numpy as np
import scipy.integrate as spInt
import scipy as sp
import matplotlib.pyplot as plt

"""  
Part (a)
"""

# Gives the nth Hermite function evaluated at x
def H(n, x):
    if n == 0:
        return 1.0
    elif n == 1:
        return 2*x
    else:
        return (2*x*H(n-1, x)) - (2*(n-1)*H(n-2, x))

# wavefunction of nth energy level of 1D quantum SHO at x
def psi_n(x, n):
    C = 1/(np.sqrt((2**n) * np.sqrt(np.pi) * m.factorial(n)))

    return C * H(n, x) * np.exp(-(x**2)/2)

nList = [0, 1, 2, 3]
xList = np.arange(-4, 4, 0.01)



for n in nList:
    psiVals = psi_n(xList, n)
    plt.plot(xList, psiVals, label=rf'$n = {n}$')

plt.xlabel('Position, x')
plt.ylabel(r'$\psi_n (x)$')
plt.legend()
plt.title('nth Wavefunction of 1D QSHO')
plt.show()



"""  
Part (b)
"""

n = 30
xList = np.arange(-10, 10, 0.01)
psiVals = psi_n(xList, n)

plt.plot(xList, psiVals)
plt.xlabel('Position, x')
plt.ylabel(r'$\psi_{30} (x)$')
plt.title('30th Wavefunction of 1D QSHO')
plt.show()



"""  
Part (c)
"""

N = 100         # number of points for Gaussian quadrature
bounds = 10      # some large number that is "infty" for out integral


def f(x):
    psiStuff = np.abs(psi_n(x, 5))**2

    return psiStuff * (x**2)

def uncertainty():
    integral = spInt.fixed_quad(f, -bounds, bounds, n=N)[0]

    return np.sqrt(integral)

print(uncertainty())


""" 
Part (d) 
"""

n = 5
N = 2*n +2 
roots, weights = sp.special.roots_hermite(N) # +2 from the x^2 that we're integrating, the 2n from |psi_n|^2

gaussHermite = 0
for i in range(N):
    gaussHermite += weights[i] * (roots[i]*np.abs(H(n, roots[i])))**2

gaussHermite = gaussHermite / (2**n * m.factorial(n) * np.sqrt(np.pi))

print(np.sqrt(gaussHermite))