import numpy as np
import matplotlib.pyplot as plt

"""  
These seem to agree with the values in https://en.wikipedia.org/wiki/Lagrange_point
"""

""" 
Still have to do part (a) 
"""

def quinticEqn(rP, mP):
    # rP: r prime
    # mP: m prime

    return (rP**5) - 2*(rP**4) + (rP**3) + (mP - 1)*(rP**2) + 2*rP - 1

def rDerivQuintic(rP, mP):

    return 5*(rP**4) - 8*(rP**3) + 3*(rP**2) + 2*(mP - 1)*(rP) + 2


def newtonStep(r, m, f, fP):
    step = f(r, m)/fP(r, m)

    return r - step

def newtonMethod(rP0, mP, tolerance = 1e-4):
    rPList = [rP0]

    rP_prev = rP0

    while True: 
        rP_curr = newtonStep(rP_prev, mP, quinticEqn, rDerivQuintic)
        rPList.append(rP_curr)

        if np.abs(rP_curr - rP_prev) < tolerance:
            break
        else:
            rP_prev = rP_curr

    return rPList


"""  
Moon and Earth
"""

m = 7.348e22
M = 5.974e24

# r here is a guess!
r = 1.0e4
R = 3.844e8

mP = m/M
rP = r/R

rPList = newtonMethod(rP, mP)

print(f'The Lagrange Point for the Moon-Earth system is located at r = {rPList[-1]*R} meters')
print(f'And, a test that this is a 0 of the appropriate function is: {quinticEqn(rP, mP)}')


"""  
Earth and Sun
"""

m = 5.974e24
M = 1.989e30

# r here is a guess!
r = 1.0e9
R = 1.5e11

mP = m/M
rP = r/R

rPList = newtonMethod(rP, mP)

print(f'The Lagrange Point for the Earth-Sun system is located at r = {rPList[-1]*R} meters')
print(f'And, a test that this is a 0 of the appropriate function is: {quinticEqn(rP, mP)}')


"""  
Jupiter-Mass Planet and Sun at Earth Distance
"""

m = 1.898e27
M = 1.989e30

# r here is a guess!
r = 1.0e7
R = 1.5e11

mP = m/M
rP = r/R

rPList = newtonMethod(rP, mP)

print(f'The Lagrange Point for the Jupiter-Sun system is located at r = {rPList[-1]*R} meters')
print(f'And, a test that this is a 0 of the appropriate function is: {quinticEqn(rP, mP)}')