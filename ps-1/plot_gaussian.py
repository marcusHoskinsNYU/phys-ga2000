import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, height, stdDev, mean): 
    # This function gives you the value of a gaussian at x
    # x: x value where we evaluate Gaussian
    # height: height of the Gaussian
    # stdDev: the standard deviation of the Gaussian
    # mean: the mean of the standard deviation

    return height * np.exp(-(x-mean)**2 / (2 * stdDev**2))

# --- ALL INPUTS ---
# Parameters given for our gaussian
height = 1
mean = 0
stdDev = 3
# Now, set up the range of our x values
xVals = np.arange(-10, 10, 0.1)

# Then, save the corresponding y values
yVals = gaussian(xVals, height, stdDev, mean)

# Now, plot the y values as a function of the x values
plt.plot(xVals, yVals, label = r"A $\exp(\frac{-\left(\mu -x)^2}{2\sigma^2}\right)$")
plt.title(r"Gaussian Plot with $\mu = 0$, $\sigma=3$, $A=1$")
plt.xlabel(r"x values, valued in $[-10, 10]$")
plt.ylabel("Gaussian values")
plt.legend()
# And then save them to a png
plt.savefig('gaussian.png')