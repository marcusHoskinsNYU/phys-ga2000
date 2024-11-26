import numpy as np
import matplotlib.pyplot as plt

"""  
Part (a)
"""

# Here, I will be using 4th order Runge-Kutta to solve these equations

# || CONSTANTS
t0 = 0
tf = 50
steps = 1000
deltaT = (tf-t0)/steps
omega = 1

# || FUNCTIONS
def f0(w0, w1, omega):
    return w1

def f1(w0, w1, omega):
    return -omega**2 * w0

def SHO_plot(t0, tf, deltaT, omega, w0_initial, w1_initial):
    # w0 = x
    # w1 = v

    # || SAVED VALUES
    tList = np.arange(t0, tf, step=deltaT)
    w0List = [] # w0 = x
    w1List = [] # w1 = v

    w0_curr = w0_initial
    w1_curr = w1_initial

    for t in tList:

        w0List.append(w0_curr)
        w1List.append(w1_curr)

        w0_prev = w0_curr
        w1_prev = w1_curr

        # k1
        k1_w0 = deltaT*f0(w0_prev, w1_prev, omega)
        k1_w1 = deltaT*f1(w0_prev, w1_prev, omega)

        # k2
        k2_w0 = deltaT*f0(w0_prev + 0.5*k1_w0, w1_prev + 0.5*k1_w1, omega)
        k2_w1 = deltaT*f1(w0_prev + 0.5*k1_w0, w1_prev + 0.5*k1_w1, omega)

        # k3
        k3_w0 = deltaT*f0(w0_prev + 0.5*k2_w0, w1_prev + 0.5*k2_w1, omega)
        k3_w1 = deltaT*f1(w0_prev + 0.5*k2_w0, w1_prev + 0.5*k2_w1, omega)

        # k4
        k4_w0 = deltaT*f0(w0_prev + k3_w0, w1_prev + k3_w1, omega)
        k4_w1 = deltaT*f1(w0_prev + k3_w0, w1_prev + k3_w1, omega)

        # w_n+1
        w0_curr = w0_prev + (1/6.)*k1_w0 + (1/3.)*k2_w0 + (1/3.)*k3_w0 +(1/6.)*k4_w0
        w1_curr = w1_prev + (1/6.)*k1_w1 + (1/3.)*k2_w1 + (1/3.)*k3_w1 +(1/6.)*k4_w1

    return tList, w0List, w1List



# || INITIAL CONDITIONS
w0_initial = 1
w1_initial = 0

tList, w0List, w1List = SHO_plot(t0, tf, deltaT, omega, w0_initial, w1_initial)

plt.plot(tList, w0List)
plt.tick_params(axis='both', labelsize=14)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Position, x [m]', fontsize=14)
plt.title(fr'Simple Harmonic Oscillator Position vs Time, $x_0=${w0_initial}, $v_0=${w1_initial}', fontsize=14)
plt.show()

plt.plot(w0List, w1List)
plt.tick_params(axis='both', labelsize=14)
plt.xlabel('Position, x [m]', fontsize=14)
plt.ylabel('Velocity, dx/dt [m/s]', fontsize=14)
plt.title(fr'Simple Harmonic Oscillator Phase Space, $x_0=${w0_initial}, $v_0=${w1_initial}', fontsize=14)
plt.show()

"""  
Part (b)
"""

plt.plot(tList, w0List, label=fr'$x_0=${w0_initial}, $v_0=${w1_initial}')

# || INITIAL CONDITIONS
w0_initial = 2
w1_initial = 0

tList, w0List, _ = SHO_plot(t0, tf, deltaT, omega, w0_initial, w1_initial)

plt.plot(tList, w0List, label=fr'$x_0=${w0_initial}, $v_0=${w1_initial}')
plt.tick_params(axis='both', labelsize=14)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Position, x [m]', fontsize=14)
plt.title(fr'Simple Harmonic Oscillator Position vs Time', fontsize=14)
plt.legend(fontsize=14)
plt.show()


"""  
Part (c)
"""

# || FUNCTIONS
def g0(w0, w1, omega):
    return w1

def g1(w0, w1, omega):
    return -(omega**2) * (w0**3)

def AHO_plot(t0, tf, deltaT, omega, w0_initial, w1_initial):

    # w0 = x
    # w1 = v

    # || SAVED VALUES
    tList = np.arange(t0, tf, step=deltaT)
    w0List = [] # w0 = x
    w1List = [] # w1 = v

    w0_curr = w0_initial
    w1_curr = w1_initial

    for t in tList:

        w0List.append(w0_curr)
        w1List.append(w1_curr)

        w0_prev = w0_curr
        w1_prev = w1_curr

        # k1
        k1_w0 = deltaT*g0(w0_prev, w1_prev, omega)
        k1_w1 = deltaT*g1(w0_prev, w1_prev, omega)

        # k2
        k2_w0 = deltaT*g0(w0_prev + 0.5*k1_w0, w1_prev + 0.5*k1_w1, omega)
        k2_w1 = deltaT*g1(w0_prev + 0.5*k1_w0, w1_prev + 0.5*k1_w1, omega)

        # k3
        k3_w0 = deltaT*g0(w0_prev + 0.5*k2_w0, w1_prev + 0.5*k2_w1, omega)
        k3_w1 = deltaT*g1(w0_prev + 0.5*k2_w0, w1_prev + 0.5*k2_w1, omega)

        # k4
        k4_w0 = deltaT*g0(w0_prev + k3_w0, w1_prev + k3_w1, omega)
        k4_w1 = deltaT*g1(w0_prev + k3_w0, w1_prev + k3_w1, omega)

        # w_n+1
        w0_curr = w0_prev + (1/6.)*k1_w0 + (1/3.)*k2_w0 + (1/3.)*k3_w0 +(1/6.)*k4_w0
        w1_curr = w1_prev + (1/6.)*k1_w1 + (1/3.)*k2_w1 + (1/3.)*k3_w1 +(1/6.)*k4_w1

    return tList, w0List, w1List


# || INITIAL CONDITIONS
w0_initial = 1
w1_initial = 0

tList, w0List, _ = AHO_plot(t0, tf, deltaT, omega, w0_initial, w1_initial)

plt.plot(tList, w0List)
plt.tick_params(axis='both', labelsize=14)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Position, x [m]', fontsize=14)
plt.title(fr'Anharmonic ($x^3$) Oscillator Position vs Time, $x_0=${w0_initial}, $v_0=${w1_initial}', fontsize=14)
plt.show()



plt.plot(tList, w0List, label=fr'$x_0=${w0_initial}, $v_0=${w1_initial}')

# || INITIAL CONDITIONS
w0_initial = 2
w1_initial = 0

tList, w0List, _ = AHO_plot(t0, tf, deltaT, omega, w0_initial, w1_initial)

plt.plot(tList, w0List, label=fr'$x_0=${w0_initial}, $v_0=${w1_initial}')


# || INITIAL CONDITIONS
w0_initial = 0.5
w1_initial = 0

tList, w0List, _ = AHO_plot(t0, tf, deltaT, omega, w0_initial, w1_initial)

plt.plot(tList, w0List, label=fr'$x_0=${w0_initial}, $v_0=${w1_initial}')

plt.tick_params(axis='both', labelsize=14)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Position, x [m]', fontsize=14)
plt.title(fr'Anharmonic ($x^3$) Oscillator Position vs Time', fontsize=14)
plt.legend(fontsize=14)
plt.show()



"""  
Part (d)
"""

# || INITIAL CONDITIONS
w0_initial = 1
w1_initial = 0

_, w0List, w1List = AHO_plot(t0, tf, deltaT, omega, w0_initial, w1_initial)

plt.plot(w0List, w1List)
plt.tick_params(axis='both', labelsize=14)
plt.xlabel('Position, x [m]', fontsize=14)
plt.ylabel('Velocity, dx/dt [m/s]', fontsize=14)
plt.title(fr'Anharmonic ($x^3$) Oscillator Phase Space, $x_0=${w0_initial}, $v_0=${w1_initial}', fontsize=14)
plt.show()


plt.plot(w0List, w1List, label=fr'$x_0=${w0_initial}, $v_0=${w1_initial}')

# || INITIAL CONDITIONS
w0_initial = 2
w1_initial = 0

_, w0List, w1List = AHO_plot(t0, tf, deltaT, omega, w0_initial, w1_initial)

plt.plot(w0List, w1List, label=fr'$x_0=${w0_initial}, $v_0=${w1_initial}')


# || INITIAL CONDITIONS
w0_initial = 0.5
w1_initial = 0

_, w0List, w1List = AHO_plot(t0, tf, deltaT, omega, w0_initial, w1_initial)

plt.plot(w0List, w1List, label=fr'$x_0=${w0_initial}, $v_0=${w1_initial}')

plt.tick_params(axis='both', labelsize=14)
plt.xlabel('Position, x [m]', fontsize=14)
plt.ylabel('Velocity, dx/dt [m/s]', fontsize=14)
plt.title(fr'Anharmonic ($x^3$) Oscillator Phase Space', fontsize=14)
plt.legend(fontsize=14)
plt.show()



"""  
Part (e)
"""

# || CONSTANTS
t0 = 0
tf = 20
steps = 10000
deltaT = (tf-t0)/steps
omega = 1

# || FUNCTIONS
def h0(w0, w1, omega, mu):
    return w1

def h1(w0, w1, omega, mu):
    return mu*(1 - w0**2)*w1 - (omega**2) * w0

def vanDerPolO_plot(t0, tf, deltaT, omega, mu, w0_initial, w1_initial):

    # w0 = x
    # w1 = v

    # || SAVED VALUES
    tList = np.arange(t0, tf, step=deltaT)
    w0List = [] # w0 = x
    w1List = [] # w1 = v

    w0_curr = w0_initial
    w1_curr = w1_initial

    for t in tList:

        w0List.append(w0_curr)
        w1List.append(w1_curr)

        w0_prev = w0_curr
        w1_prev = w1_curr

        # k1
        k1_w0 = deltaT*h0(w0_prev, w1_prev, omega, mu)
        k1_w1 = deltaT*h1(w0_prev, w1_prev, omega, mu)

        # k2
        k2_w0 = deltaT*h0(w0_prev + 0.5*k1_w0, w1_prev + 0.5*k1_w1, omega, mu)
        k2_w1 = deltaT*h1(w0_prev + 0.5*k1_w0, w1_prev + 0.5*k1_w1, omega, mu)

        # k3
        k3_w0 = deltaT*h0(w0_prev + 0.5*k2_w0, w1_prev + 0.5*k2_w1, omega, mu)
        k3_w1 = deltaT*h1(w0_prev + 0.5*k2_w0, w1_prev + 0.5*k2_w1, omega, mu)

        # k4
        k4_w0 = deltaT*h0(w0_prev + k3_w0, w1_prev + k3_w1, omega, mu)
        k4_w1 = deltaT*h1(w0_prev + k3_w0, w1_prev + k3_w1, omega, mu)

        # w_n+1
        w0_curr = w0_prev + (1/6.)*k1_w0 + (1/3.)*k2_w0 + (1/3.)*k3_w0 +(1/6.)*k4_w0
        w1_curr = w1_prev + (1/6.)*k1_w1 + (1/3.)*k2_w1 + (1/3.)*k3_w1 +(1/6.)*k4_w1

    return tList, w0List, w1List

# || INITIAL CONDITIONS
w0_initial = 1
w1_initial = 0

mu = 1
_, w0List, w1List = vanDerPolO_plot(t0, tf, deltaT, omega, mu, w0_initial, w1_initial)
plt.plot(w0List, w1List, label=fr'$\mu=${mu}')

mu = 2
_, w0List, w1List = vanDerPolO_plot(t0, tf, deltaT, omega, mu, w0_initial, w1_initial)
plt.plot(w0List, w1List, label=fr'$\mu=${mu}')

mu = 4
_, w0List, w1List = vanDerPolO_plot(t0, tf, deltaT, omega, mu, w0_initial, w1_initial)
plt.plot(w0List, w1List, label=fr'$\mu=${mu}')

plt.tick_params(axis='both', labelsize=14)
plt.xlabel('Position, x [m]', fontsize=14)
plt.ylabel('Velocity, dx/dt [m/s]', fontsize=14)
plt.title(fr'van der Pol Oscillator Phase Space, $x_0=${w0_initial}, $v_0=${w1_initial}, $\omega=${omega}', fontsize=14)
plt.legend(fontsize=14)
plt.show()
