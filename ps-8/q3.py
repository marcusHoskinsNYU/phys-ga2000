import numpy as np
import matplotlib.pyplot as plt
import math

"""  
Part (a)
"""

y = np.loadtxt('dow.txt', float)
lenY = len(y)
x = np.arange(lenY)

plt.plot(x, y)
plt.ylabel('EOD Dow Closing Value[$]')
plt.xlabel('Business Day')
plt.title('Dow Jones Closing Value per Business Day')
plt.show()


"""  
Part (b)
"""

coeff = np.fft.rfft(y)


"""  
Part (c)
"""

percent = 0.1
keepNum = math.ceil(percent*lenY)
keptCoeff = np.zeros(lenY)

for i in range(keepNum):
    keptCoeff[i] = coeff[i]


"""  
Part (d)
"""

invCoeff = np.fft.irfft(keptCoeff, lenY)
plt.plot(x, y, label='Original')
plt.plot(x, invCoeff, label=f'Inverted Coefficients, {percent*100}%')
plt.ylabel('EOD Dow Closing Value[$]')
plt.xlabel('Business Day')
plt.title('Dow Jones Closing Value per Business Day')
plt.legend()
plt.show()


"""  
Part (e)
"""

percent = 0.02
keepNum = math.ceil(percent*lenY)
keptCoeff = np.zeros(lenY)

for i in range(keepNum):
    keptCoeff[i] = coeff[i]


invCoeff = np.fft.irfft(keptCoeff, lenY)
plt.plot(x, y, label='Original')
plt.plot(x, invCoeff, label=f'Inverted Coefficients, {percent*100}%')
plt.ylabel('EOD Dow Closing Value[$]')
plt.xlabel('Business Day')
plt.title('Dow Jones Closing Value per Business Day')
plt.legend()
plt.show()
