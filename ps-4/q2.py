import scipy.integrate as spInt
import numpy as np
import matplotlib.pyplot as plt

# || CONSTANTS
power = 4           # Power of our V(x) = x^power
m = 1               # Value of mass
N = 20              # Number of points used in Gaussian quadrature


# || FUNCTIONS

# our V(x)
def V(x, p):
    return x**p

# The actual function being integrated
def f(x, a, p):
    return 1/np.sqrt(V(a, p) - V(x, p))


# period of oscillation
def T(a, mass):
    integral = spInt.fixed_quad(f, 0, a, args=(a, power), n = N)[0]

    return np.sqrt(8*mass) * integral


"""
Part (b)
"""

aList = np.arange(0, 2, 0.01)
tList = [T(a, m) for a in aList]

plt.plot(aList, tList)
plt.xlabel('Initial amplitude, a [m]')
plt.ylabel('Period, T [s]')
plt.title(rf'Period vs Initial Amplitude for $V(x)=x^{power}$')
plt.show()
