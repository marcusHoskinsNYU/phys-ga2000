import numpy as np


""" Part 1 """

print( np.float32(1) + np.float32(10**(-7))) # any lower than 10^-7 this prints 1.0
print( np.float64(1) + np.float64(10**(-15))) # any lower than 10^-15 this prints 1.0



""" Part 2 """

print(np.float32(1.7014118e-45)) # I get underflow (ie a 0 value) for -46 or less
print(np.float64(2**(-1075))) # I get underflow for -1076 or less

print(np.float32(1.7014118e38)) # If I go to e39, I get overflow. Eqvuivalently, this is 2**127
print(np.float64(2**1023)) # going to 1024 gives error
