import numpy as np
import matplotlib.pyplot as plt


dataFilePath = 'ps-5\signal.dat'


"""  
Part (a)
"""

timeList = []
signalList = []

i = 0
with open(dataFilePath, 'r') as file:
    for line in file:
        line = line.strip().split('|')
        if i != 0:
            timeList.append(float(line[1]))
            signalList.append(float(line[2]))

        i += 1

plt.scatter(timeList, signalList)
plt.ylabel('Signal data')
plt.xlabel('Time data, t [s]')
plt.title('Signal vs Time Data')
plt.show()

timeList = np.array(timeList)
signalList = np.array(signalList)

"""  
Part (b)
"""

# Need to rescale t, as it gives a 0 singular value in w
tBar = np.mean(timeList)
sigmaT = np.std(timeList)
newTimeList = (timeList - tBar)/sigmaT

A = np.zeros((len(newTimeList), 4))
A[:, 0] = 1.
A[:, 1] = newTimeList
A[:, 2] = newTimeList**2
A[:, 3] = newTimeList**3

(u, w, vt) = np.linalg.svd(A, full_matrices=False)
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
x = ainv.dot(signalList)
bm = A.dot(x)

print(x)

plt.plot(newTimeList, signalList, 'k.', label='data')
plt.plot(newTimeList, bm, 'r.', label='model')
plt.ylabel('Signal data')
plt.xlabel(r'Rescaled Time Data, $(t-\mu_t)/\sigma_t$')
plt.title('Signal vs Rescaled Time')
plt.legend()
plt.show()

"""  
Part (c)
"""

print(np.mean(np.abs(signalList - bm)))

plt.plot(newTimeList, signalList - bm, 'k.', label='data - model')
plt.xlabel('Rescaled Time')
plt.ylabel('Residual, data - model')
plt.legend()
plt.title('Residuals vs Rescaled Time')
plt.show()


""" 
Part (d)
"""
order = 50 

A = np.zeros((len(newTimeList), order+1))
A[:, 0] = 1.
for i in range(1, order+1):
    A[:, i] = newTimeList**i

(u, w, vt) = np.linalg.svd(A, full_matrices=False)
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
x = ainv.dot(signalList)
bm = A.dot(x)

print(x)

plt.plot(newTimeList, signalList, 'k.', label='data')
plt.plot(newTimeList, bm, 'r.', label=f'model, order={order}')
plt.ylabel('Signal data')
plt.xlabel(r'Rescaled Time Data, $(t-\mu_t)/\sigma_t$')
plt.title('Signal vs Rescaled Time')
plt.legend()
plt.show()


"""  
Part (e)
"""


orderFreq = 50
timeSpan = (np.max(newTimeList) - np.min(newTimeList))

A = np.zeros((len(newTimeList), (2*orderFreq)+1))
A[:, 0] = 1. # zero-point offset
for i in range(1, orderFreq):
    freq = (i+1) * np.pi / timeSpan
    A[:, 2*i -1] = np.cos(freq * newTimeList)
    A[:, 2*i] = np.sin(freq * newTimeList)

freq = (orderFreq) * np.pi / timeSpan
A[:, 2*orderFreq -1] = np.cos(freq * newTimeList)
A[:, 2*orderFreq] = np.sin(freq * newTimeList)

print(A)
(u, w, vt) = np.linalg.svd(A, full_matrices=False)
ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
x = ainv.dot(signalList)
bm = A.dot(x)

print(x)

plt.plot(newTimeList, signalList, 'k.', label='data')
plt.plot(newTimeList, bm, 'r.', label=f'model, sin-cos')
plt.ylabel('Signal data')
plt.xlabel(r'Rescaled Time Data, $(t-\mu_t)/\sigma_t$')
plt.title('Signal vs Rescaled Time')
plt.legend()
plt.show()

print(np.mean(np.abs(signalList - bm)))

plt.plot(newTimeList, signalList - bm, 'k.', label='data - model')
plt.xlabel('Rescaled Time')
plt.ylabel('Residual, data - model')
plt.legend()
plt.title('Residuals vs Rescaled Time')
plt.show()