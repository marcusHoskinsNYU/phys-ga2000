import numpy as np
import matplotlib.pyplot as plt

# Constants
NTl = 1000              # Number of thallium atoms
NPb =0                  # Number of lead atoms
tau = 3.053*60          # Half life of thallium in seconds
h = 1                 # time step
mu = np.log(2)/tau      # mu used in transformation method
tmax = 1000             # Total time

# exponentially distributed random numbers
def xFunc(z):
    # z is uniformly distributed number from 0 to 1
    return (-1/mu)*np.log(1-z)

zList = np.random.rand(tmax)
xList = xFunc(zList)
xListSorted = np.sort(xList).astype(int)

tpoints = np.arange(0, xListSorted[-1]+10, h)
NTlList = np.array([NTl])
NTlList = np.repeat(NTlList, len(tpoints))

for i in range(len(xList)):

    NTlList[xListSorted[i]:] -= 1

plt.plot(tpoints, NTlList)
plt.xlabel('Time (s)')
plt.ylabel('Number of undecayed Tl atoms')
plt.title('Tl number vs time')
plt.show()