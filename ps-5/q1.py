
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import math

def f(x):
    return 1 + 0.5*np.tanh(2*x)

def f_jax(x):
    return 1 + 0.5*jnp.tanh(2*x)

def central_diff(func, x, dx=1e-5):
    return (func(x + (dx/2)) - func(x - (dx/2)))/dx

def fPrimeAnalytic(x):
    return 1 - (np.tanh(2*x))**2

xApproxList = np.arange(-2, 2, 0.1)
xApproxJaxList = jnp.linspace(-2, 2, 40)
xAnalList = np.arange(-2, 2, 0.01)
fPrimeApproxList = central_diff(f, xApproxList)
fPrimeAnalList = fPrimeAnalytic(xAnalList)
fPrimeJax = jax.vmap(jax.grad(f_jax))(xApproxJaxList)

plt.plot(xApproxList, fPrimeApproxList, 'k.', label='Central Difference Approx')
plt.plot(xAnalList, fPrimeAnalList, 'r-', label='Analytic')
plt.plot(xApproxJaxList, fPrimeJax, 'b.', label='Jax autodiff')
plt.xlabel('x')
plt.ylabel(r'$\frac{df}{dx}$')
plt.title('Derivative of f(x) vs x')
plt.legend()
plt.show()
