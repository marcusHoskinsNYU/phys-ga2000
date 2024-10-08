import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spInt

"""  
Part (a)
"""

def integrand(x, a):
    return (x**(a-1))*np.exp(-x)

xList = np.arange(0, 5, 0.01)
a2List = integrand(xList, 2)
a3List = integrand(xList, 3)
a4List = integrand(xList, 4)

plt.plot(xList, a2List, label='a=2')
plt.plot(xList, a3List, label='a=3')
plt.plot(xList, a4List, label='a=4')
plt.xlabel('x')
plt.ylabel(r'$x^{a-1}e^{-x}$')
plt.title('Integrand of Gamma Function vs x')
plt.legend()
plt.show()


"""  
Part (e)
"""

def changeOfVar(z, a):
    c = a-1
    return (z*c)/(1-z)

def newIntegrand(z, a):

    c = a-1

    jacobian = ((c+changeOfVar(z, a))**2)/c

    return jacobian * np.exp(c*np.log(changeOfVar(z, a)) - changeOfVar(z, a))

def gamma(a):
    return spInt.fixed_quad(newIntegrand, 0., 1., args=[a], n=20)[0]
    
print(gamma(1.5))


"""  
Part (f)
"""

print(gamma(3))
print(gamma(6))
print(gamma(10))