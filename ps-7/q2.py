import numpy as np
import scipy.optimize as spOpt

def y(x):
    # this is the function for our problem
    return ((x-0.3)**2)*np.exp(x)

def brentRoot(f, a, b, c):
    # f(x) = alpha x^2 + beta x + gamma
    min = b
    
    epsilon = 1.e-10

    fa = f(a)
    fb = f(b)
    fc = f(c)

    numerator = (((b-a)**2) * (fb - fc)) - (((b-c)**2) * (fb - fa))
    denominator = 2 * (((b-a) * (fb - fc)) - ((b-c) * (fb - fa)))

    return min - (numerator/(denominator+epsilon))


def goldenSearch(func, a0, b0, c0, tol=1.e-10):
    # Taken from lecture
    numGolden = 0
    gsection = (3. - np.sqrt(5)) / 2
    a = a0
    b = b0
    c = c0
    while(np.abs(c - a) > tol):
        # Split the larger interval
        if((b - a) > (c - b)):
            x = b
            b = b - gsection * (b - a)
        else:
            x = b + gsection * (c - b)
        fb = func(b)
        fx = func(x)
        if(fb < fx):
            c = x
        else:
            a = b
            b = x 
        numGolden += 1

    print(f'Number of Golden Search Runs was {numGolden}')
    return b


def brent1D(func, a0, b0, c0, tol=1.0e-8):

    # assumes that a < b < c, and further that f(b) < f(c) and f(b) < f(a)
    numBrent = 0

    a = a0
    b = b0
    c = c0

    fa = func(a)
    fb = func(b)
    fc = func(c)

    u = brentRoot(func, a, b, c)
    if (u < a) or (u > c):
        print(f'Number of Parabola Fit Runs was {numBrent}')
        return goldenSearch(func, a, b, c)
    
    lastStep = np.abs(b - u)
    
    fu = func(u)
    if (fu < fb):
        x = u
        w = b
        if u < b:
            c = w
        else:
            a = w
        b = u
    else:
        x = b
        w = u
        if b < u:
            c = w
        else:
            a = w


    stepBeforeLast = 1.e10
    numBrent += 1

    while (np.abs(c - a) > tol):

        # make a parabolic fit and find minimum
        u = brentRoot(func, a, b, c)

        # break to golden section search if conditions are met
        if (u < a) or (u > c) or (np.abs(b - u) > stepBeforeLast):
            print(stepBeforeLast, u, a, b, c)
            print(f'Number of Parabola Fit Runs was {numBrent}')
            return goldenSearch(func, a, b, c)
        
        fu = func(u)
        
        stepBeforeLast = lastStep
        lastStep = np.abs(b - u)
        
        v = w
        if (fu < fb):
            x = u
            w = b
            if u < b:
                c = w
            else:
                a = w
            b = u
        else:
            x = b
            w = u
            if b < u:
                c = w
            else:
                a = w

        numBrent += 1
    
    
    print(f'Number of Parabola Fit Runs was {numBrent}, and there were no golden search runs')
    return x


a = 0
b = 0.5
c = 2

print(brent1D(y, a, b, c))
print(spOpt.brent(y, brack=(a, b, c), tol=1.e-10))