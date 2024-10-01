import scipy.integrate as spInt
import numpy as np
import matplotlib.pyplot as plt

# || CONSTANTS
V = 1000 / (100*100*100)    # Colume of solid
rho = 6.022e28              # number density of aluminum
thetaD = 428                # Debye temperature
kB = 1.380649e-23           # Boltzmann constant
N = 50                      # Number of sample points

# || FUNCTIONS

# Function being integrated
def f(x):
    return (np.exp(x) * (x**4))/((np.exp(x) - 1)**2)


# Actual function desired for part (a)
def cv(T, order):
    integral = spInt.fixed_quad(f, 0.0, thetaD/T, n=order)[0]
    return 9*V*rho*kB * ((T/thetaD)**3) * integral


print()

""" 
Part (b) 
"""

Tlist = np.arange(5, 500, 1)
cvList = np.array([cv(T, N) for T in Tlist])

plt.plot(Tlist, cvList)
plt.xlabel('Temperature, T [K]')
plt.ylabel(r'Heat Capacity, $C_V$ [J/K]')
plt.title(rf'$C_V$ vs T for N={N} using Gaussian Quadrature')
plt.show()


""" 
Part (c)
"""

Nlist = [1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60, 70]

for Ntemp in Nlist:
    cvList = np.array([cv(T, Ntemp) for T in Tlist])
    plt.plot(Tlist, cvList, label=f'N = {Ntemp}')

plt.xlabel('Temperature, T [K]')
plt.ylabel(f'$C_V$')
plt.title(f'$C_V$ vs T for various N values')
plt.legend()
plt.show()

"""  
Note: Seems like they are all the same? So does this mean convergence occurs for many N? Seems like after 3 or 4 it matches, which kind of makes sense because of the x^4 in f. But what about the x's from e^x?? Maybe it gets deleted by the 1/(e^x - 1)^2, as, since they're both of infinite degree, it cancels. 
"""