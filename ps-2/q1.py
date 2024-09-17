import numpy as np

mantissa = [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
total = 1
for i in range(23):
    total += (mantissa[i] * 2**(23-i))

print(total * 2**(-17))


# Notes: This actual numer is 100.98761749267578, which differs from 100.98763 by a factor of 10^-5, which is pretty good, considering we have used all digits in mantissa