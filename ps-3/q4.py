import numpy as np
import matplotlib.pyplot as plt

N = 1000 # amount of x's used in each y
Ndistrib = 1000 # amount of y's that we calculate
Nbins = 20 # number of bins

yList = []

for i in range(Ndistrib):
    xList = np.random.exponential(size=N+1)
    y_i = xList.sum() / N

    yList.append(y_i)

plt.hist(yList, bins=Nbins)
plt.xlabel(r'Value of y = $N^{-1} \sum_{i=0}^N x_i$')
plt.ylabel('Number of values in bin')
plt.title(f'Histogram of N={Ndistrib} y vals with N={N} x_i')
plt.show()




def mean(N):
    return (N+1)/N

def var(N):
    return (1 - (2/N) -(1/N**2))*(N+1)/(N**2)

def SD(N):
    return np.sqrt(var(N))

def expecP(p, N_x, N_dist=1000):
    # returns the expectation value of y^p, where we have N_dist values of y with N_x x's
    yList = []

    for i in range(N_dist):
        xList = np.random.exponential(size=N_x+1)
        y_i = xList.sum() / N_x

        yList.append(y_i**p)

    yList = np.array(yList)

    return yList.sum() / N_dist
    
def skew(N):
    mu = mean(N)
    sigma = SD(N)
    return (expecP(3, N) - mu*(sigma**2) - mu**3)/(sigma**3)

def kurt(N):
    mu = mean(N)
    sigma = SD(N)
    return (expecP(4, N) + 6*(mu**2)*(sigma**2) - 4*mu*expecP(3, N) + 3*(mu**4))/(sigma**4)


Nlist = np.arange(1, 1000, 1)

meanList = []
varList = []
skewList = []
kurtList = []

for N in Nlist:
    if (N % 100) == 0:
        print(f'N={N} Completed')
    meanList.append(mean(N))
    varList.append(var(N))
    skewList.append(skew(N))
    kurtList.append(kurt(N))

print(np.average(kurtList))
print(np.average(skewList))

figure, axis = plt.subplots(2, 2)

# For mean
axis[0, 0].plot(Nlist, meanList)
axis[0, 0].set(xlabel='N', ylabel='Mean')
axis[0, 0].set_title("Mean vs N")

# For variance
axis[0, 1].plot(Nlist, varList)
axis[0, 1].set(xlabel='N', ylabel='Variance')
axis[0, 1].set_title("Variance vs N")

# For skewness
axis[1, 0].plot(Nlist, skewList)
axis[1, 0].set(xlabel='N', ylabel='Skewness')
axis[1, 0].set_title("Skewness vs N")

# For kurtosis
axis[1, 1].plot(Nlist, kurtList)
axis[1, 1].set(xlabel='N', ylabel='Kurtosis')
axis[1, 1].set_title("Kurtosis vs N")

# Combine all the operations and display
plt.show()

kurtOne = kurtList[0]
skewOne = skewList[0]

for i in range(1, len(Nlist)):
    if kurtList[i] < 0.01*kurtOne:
        print(Nlist[i])

for i in range(1, len(Nlist)):
    if skewList[i] < 0.01*skewOne:
        print(Nlist[i])
