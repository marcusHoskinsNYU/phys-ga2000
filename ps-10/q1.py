import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import scipy as sp

"""  
Part (a)
"""


def initialPsiAtX(x, L, t0 = 0):
    # initial form of wavefunction at position x
    x0 = L/2.
    sigma = 1e-10
    kappa = 5e10
    psi0 = np.exp(-((x - x0)**2) / (2 * sigma**2)) * np.exp(1j * kappa * x)

    return psi0

def initialPsi(a, L, N):
    # computes the total initial psi for our problem
    psi0List = []
    xcurr = 0

    for i in range(N):
        psi0List.append(initialPsiAtX(xcurr, L))

        xcurr += a
    
    return psi0List

def xPosList(a, N):
    # returns the position value of the ith spatial slice of our lattice
    # i ranges from 0 to N-1
    xList = [i * a for i in range(N)]
    return xList

def A_mat(h, a, hbar, m, N):
    # construct the A matrix needed for Crank-Nicolson, as it is needed in the banded.py function, where N is the number of spatial steps we take in our box
    a1 =  1 + h * ((1j * hbar) / (2*m*(a**2)))
    a2 = -h * ((1j * hbar) / (4*m*(a**2)))

    up = 1
    down = 1
    A = np.zeros((1 + up + down, N), dtype=complex)
    
    # A = np.zeros((N, N), dtype=complex)


    """ A[0, 0] = a1
    A[0, 1] = a2

    for i in range(1, N-1):
        A[i, i-1] = a2
        A[i, i] = a1
        A[i, i+1] = a2

    A[N-1, N-2] = a2
    A[N-1, N-1] = a1 """

    # upper / lower diagonals:
    for i in range(N-1):
        A[0, 1+i] = a2
        A[2, N-2 - i] = a2
        A[1, i] = a1
    
    A[1, N-1] = a1

    return A

def vFromB(psi_curr, h, a, hbar, m):
    b1 = 1 - h * ((1j * hbar) / (2*m*(a**2)))
    b2 = h * ((1j * hbar) / (4*m*(a**2)))
    length = len(psi_curr)
    v = [(b1 * psi_curr[0]) + (b2*(psi_curr[1]))]

    for i in range(1, length-1):
        v.append((b1 * psi_curr[i]) + (b2*(psi_curr[i+1] + psi_curr[i-1])))

    v.append((b1 * psi_curr[length-1]) + (b2*(psi_curr[length-2])))
    v = np.array(v)

    return v

def solveForX(A, v):
    # solves the linear system A * x = v, where A is tridiagonal and symmetric
    up = 1
    down = 1
    x = sp.linalg.solve_banded((up, down), A, v)
    """ A = np.array(A)
    v = np.array(v)
    x = np.linalg.solve(A, v) """
    return x

def crankNicolsonStep(psi_curr, A, h, a, hbar, m):
    # takes a single time step using the crank-nicolson method to solve the Schrodinger equation in 1D
    v = vFromB(psi_curr, h, a, hbar, m)
    x = solveForX(A, v)
    newPsi = x

    return newPsi

def crankNicolsonFull(steps, h, a, L, hbar, m, N):
    # solves the Schrodinger solution for 'steps' time steps of size h using the crank-nicolson method

    A = A_mat(h, a, hbar, m, N)

    psiInitial = initialPsi(a, L, N)
    psiList = [psiInitial]
    
    psiCurr = psiInitial

    for i in range(steps):
        psiCurr = crankNicolsonStep(psiCurr, A, h, a, hbar, m)
        psiList.append(psiCurr)

    return psiList




"""  
Part (b)
"""

# || CONSTANTS
M = 9.109e-31       # mass of electron in kg
L = 1e-8            # length of box in m
t0 = 0              # initial time 
N = 1000            # number of spatial slices
a = L / N           # spacing of aptial grid points
h = 1e-18           # time step in s
hbar = 1.054e-34    # hbar
steps = 100000       # number of time steps taken
scale = 1e-9        # scale of the wavefunction values so they appear reasonable on the screen
nframes = 3

x = np.linspace(0, L, N)
psiList = crankNicolsonFull(steps, h, a, L, hbar, M, N)

newSteps = int(steps / nframes)

for i in range(nframes+1):
    plt.plot(x/scale, np.real(psiList[(i)*newSteps]), linewidth=2)
    plt.xlabel('Position, x [nm]', size=20)
    plt.ylabel(r'Re($\psi (x, t)$)', size=20)
    plt.title(rf'Re($\psi (x, t)$) vs Position, at t = {h*((i)*newSteps):.2E} s', size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

""" fig, ax = plt.subplots()
ax.set_xlim((450., 650.))
ax.set_ylim((-0.03, 0.15))

line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return (line,)

q = np.zeros((steps, N))
x = np.arange(N)
psiList = crankNicolsonFull(steps, h, a, L, hbar, M, N)

for i in range(steps):
    q[i, :] = np.real(psiList[i])

def frame(i):
    line.set_data(x, q[i, :])
    return (line,)

anim = animation.FuncAnimation(fig, frame, init_func=init, frames = steps, blit=True)

HTML(anim.to_html5_video())
 """
