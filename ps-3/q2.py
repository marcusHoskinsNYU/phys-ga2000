import numpy as np
import matplotlib.pyplot as plt

# Constants
NBi213 = 10000                  # Number of Bismuth 213 atoms
NBi209 = 0                      # Number of Bismuth 209 atoms
NTl209 = 0                      # Number of Tl209 atoms
NPb209 = 0                      # Number of Pb209 atoms
tauBi213 = 46*60                # Half life of Bi213 in seconds
tauTl209 = 2.2*60               # Half life of Tl209 in seconds
tauPb209 = 3.3*60               # Half life of Pb209 in seconds
h = 1.0                         # Size of time-step in seconds
pBiTOPb = 0.9791                # Prob of decay of Bi213 to Pb209
pBiTOTl = 0.0209                # Prob of decay of Bi213 to Tl209
pPb209 = 1 - 2**(-h/tauPb209)   # Prob of decay of Pb209 in one step
pTl209 = 1 - 2**(-h/tauTl209)   # Prob of decay of Tl209
pBi213 = 1 - 2**(-h/tauBi213)   # Prob of decay of Bi213

tmax = 20000                    # Total time

# List of plot points
tpoints = np.arange(0.0, tmax, h)
Bi213points = []
Tl209points = []
Pb209points = []
Bi209points = []

# Main loop
for t in tpoints:
    Bi213points.append(NBi213)
    Tl209points.append(NTl209)
    Pb209points.append(NPb209)
    Bi209points.append(NBi209)

    # So as not to double count, you must work up the chain, not down

    # Pb decay
    PbDecay = 0
    for i in range(NPb209):
        if np.random.rand() < pPb209:
            PbDecay += 1

    NPb209 -= PbDecay
    NBi209 += PbDecay


    # Tl decay
    TlDecay = 0
    for i in range(NTl209):
        if np.random.rand() < pTl209:
            TlDecay += 1

    NTl209 -= TlDecay
    NPb209 += TlDecay


    # Bi213 decay
    BiTOPbDecay = 0
    BiTOTlDecay = 0
    for i in range(NBi213):

        if np.random.rand() < pBi213:
            # First, BiTl decay
            randNum = np.random.rand()
            if randNum < pBiTOTl:
                BiTOTlDecay += 1
            else:
                BiTOPbDecay += 1


    NTl209 += BiTOTlDecay
    NPb209 += BiTOPbDecay
    NBi213 -= (BiTOPbDecay + BiTOTlDecay)

# Make graph
plt.plot(tpoints, Bi213points, 'b', label='Bi213')
plt.plot(tpoints, Tl209points, 'g', label='Tl209')
plt.plot(tpoints, Pb209points, 'r', label='Pb209')
plt.plot(tpoints, Bi209points, 'm', label='Bi209')
plt.xlabel("Time (s)")
plt.ylabel('Number of atoms')
plt.legend()
plt.title('Newman Exercise 10.2 Decay Chain')
plt.show()

plt.plot(tpoints, Tl209points, 'g')
plt.xlabel("Time (s)")
plt.ylabel('Number of atoms')
plt.title('Tl209 Number')
plt.show()

plt.plot(tpoints, Pb209points, 'r')
plt.xlabel("Time (s)")
plt.ylabel('Number of atoms')
plt.title('Pb209 Number')
plt.show()
