import numpy as np
import matplotlib.pyplot as plt

"""  
Part (a)
"""

def aProgram(fileName, instrument):
    y = np.loadtxt(fileName, float)
    x = np.arange(len(y))
    
    coeff = np.fft.fft(y)
    xTransform = np.arange(0, 10000)
    
    plt.plot(x, y)
    plt.ylabel('Input Value')
    plt.xlabel('Input Number')
    plt.title(f'Input Data for {instrument}')
    plt.show()

    plt.plot(xTransform, np.abs(coeff)[0:10000])
    plt.ylabel('Coefficient Value')
    plt.xlabel('Coefficient')
    plt.title(f'Transform Data for {instrument}')
    plt.show()
    

aProgram('piano.txt', 'Piano')
aProgram('trumpet.txt', 'Trumpet')
