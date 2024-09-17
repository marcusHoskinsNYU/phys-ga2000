import numpy as np
from datetime import datetime

L = 200
startTime = datetime.now()

# With for loop
M = 0
for i in range(-L, L+1): 
    for j in range(-L, L+1):
        for k in range(-L, L+1):
            if (i != 0) or (j != 0) or (k != 0):
                M += 1/np.sqrt(i**2 + j**2 + k**2)

print("With for loop, Madelung constant is ", M, " in ", (datetime.now()-startTime))



# Without for loop 
startTime = datetime.now()

i = np.arange(-L, L+1)
j = np.arange(-L, L+1)
k = np.arange(-L, L+1)

I,J,K = np.meshgrid(i, j, k, indexing='ij')

distSquared = I**2 + J**2 + K**2
with np.errstate(divide='ignore', invalid='ignore'):
    invDist = np.where(distSquared != 0, 1 / np.sqrt(distSquared), 0)

M = np.sum(invDist)

print("Without for loop, Madelung constant is ", M, " in ", (datetime.now()-startTime))
