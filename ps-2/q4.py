import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Ngrid = 100
Niter = 100

""" xList = np.arange(4, step = 1/Ngrid) - 2
yList = (np.arange(4, step = 1/Ngrid) - 2)

X, Y = cMat = np.meshgrid(xList, yList, indexing='ij')
z = 0

for i in range(Niter):
    zP = z**2 + (X + (0+1j)*Y)
    if ((zP.real)**2 + (zP.imag)**2).any() > 2:
        X = 0
        Y = 0
    
    z = zP



df = pd.DataFrame({'x': plottedX, 'y': plottedY})

# Small bins
plt.hist2d(df.x, df.y, bins=(600, 600), cmap=plt.cm.Greys_r)
plt.show() """


# For loop version

xList = np.arange(4, step = 1/Ngrid) - 2
yList = (np.arange(4, step = 1/Ngrid) - 2)

plottedX = []
plottedY = []

for x in xList:
    for y in yList:
        withinTwo = True
        z = 0
        for i in range(Niter):
            zP = z**2 + (x + y*(0+1j))
            if (zP.real)**2 + (zP.imag)**2 > 2:
                withinTwo = False
                break
            else: 
                z = zP
        if withinTwo: 
            plottedX.append(x)
            plottedY.append(y)

df = pd.DataFrame({'x': plottedX, 'y': plottedY})

# Small bins
plt.hist2d(df.x, df.y, bins=(600, 600), cmap=plt.cm.gray_r)
plt.title(f'Mandelbrot Set, N = {Ngrid} ')
plt.xlabel("x = Re(c, [x] = 1")
plt.ylabel('y = Im(c), [y] = 1')
plt.show()