import numpy as np

# Part (a)

def solnA(a, b, c): 
    # ax^2 + bx + c= 0
    return np.array([(-b + np.sqrt(b**2 - 4*a*c))/(2*a), (-b - np.sqrt(b**2 - 4*a*c))/(2*a)])




# Part (b) 

def solnB(a, b, c):
    return np.array([2*c/(-b - np.sqrt(b**2 - 4*a*c)), 2*c/(-b + np.sqrt(b**2 - 4*a*c))])




# Part (c)
""" 
The issue is that we want to multiply numbers of the same order. That is, we get awful rounding errors when multiplying 1 big and 1 small together, but not as much when we multiply 2 big or 2 small
"""

def quadratic(a, b, c, closeness = 1.e2): 
    # Closeness is a measure of how many order of magnitude there are between between the numerator and 1/denominator before we switch to other method
    disc = b**2 - 4*a*c
    delta = np.sqrt(disc)

    answerList = []

    if ((np.abs(-b + delta) < 1/closeness) and (np.abs(a) > closeness)) or ((np.abs(-b + delta) > closeness) and (np.abs(a) < 1/closeness)):
        answerList.append(solnA(a, b, c)[0])
    else: 
        answerList.append(solnB(a, b, c)[0])

    if ((np.abs(-b - delta) < 1/closeness) and (np.abs(a) > closeness)) or ((np.abs(-b - delta) > closeness) and (np.abs(a) < 1/closeness)):
        answerList.append(solnA(a, b, c)[1])
    else: 
        answerList.append(solnB(a, b, c)[1])

    return np.array(answerList)


