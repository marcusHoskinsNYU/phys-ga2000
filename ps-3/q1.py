import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

NList = [30, 40,  50, 80, 100, 200, 300]


# ||    EXPLICIT MATRIX MULTIPLICATION

timeList = []

for N in NList:
    C = np.zeros([N, N], float)
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)

    startTime = datetime.now()
    for i in range(N):
        for j in range(N): 
            for k in range (N):
                C[i, j] += A[i, k]*B[k, j]
    t = (datetime.now() - startTime).total_seconds()
    timeList.append(t)
    print(f'Done with N={N} in {t:.5f}')

# Here we take logs of the equation t = # * N^e to get log(t) = log(#) + e log(N) = m*log(N) + b 

logNList = [np.log(N) for N in NList]
logTimeList = [np.log(t) for t in timeList]

m, b = np.polyfit(logNList, logTimeList, 1)

fittedTimes = [np.exp(b)*(N**(m)) for N in NList]


plt.plot(NList, timeList, 'k.',label='data')
plt.plot(NList, fittedTimes, 'r-', label=f'fit: {np.exp(b):.6f}N^{m:.3f}')
plt.xlabel("N values")
plt.ylabel('Time of Multiplication (s)')
plt.title('Explicit Matrix Multiplication')
plt.legend()
plt.show()



# || DOT() METHOD MATRIX MULTIPLICATION

timeList = []

for N in NList:
    C = np.zeros([N, N], float)
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)

    startTime = datetime.now()
    C = np.dot(A, B)
    t = (datetime.now() - startTime).total_seconds()
    timeList.append(t)
    print(f'Done with N={N} in {t:.20f}')



plt.plot(NList, timeList, 'k.',label='data')
# plt.plot(fitN, fitTime, 'r-', label=r'fit: 2 N^3')
plt.xlabel("N values")
plt.ylabel('Time of Multiplication (s)')
plt.title('dot() Method Matrix Multiplication')
plt.legend()
plt.show()
