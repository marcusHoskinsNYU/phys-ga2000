import numpy as np
import matplotlib.pyplot as plt
import math

# || CONSTANTS
t0 = 0
tf = 50
steps = 1000
deltaT = (tf-t0)/steps
R = 0.08
rho = 1.22
C = 0.47
m = 1


# || FUNCTIONS
def f0(w2, w3, R, rho, C, m, g = 9.81):
    return w2

def f1(w2, w3, R, rho, C, m, g = 9.81):
    return w3

def f2(w2, w3, R, rho, C, m, g = 9.81):
    return -((np.pi * (R**2) * rho * C) / (2*m)) * w2 * np.sqrt((w2**2) + (w3**2)) 

def f3(w2, w3, R, rho, C, m, g = 9.81):
    return -g - ((np.pi * (R**2) * rho * C) / (2*m)) * w3 * np.sqrt((w2**2) + (w3**2)) 

def cannonBallDrag_plot(t0, tf, deltaT, R, rho, C, m, w0_initial, w1_initial, w2_initial, w3_initial):
    # w0 = x
    # w1 = y
    # w2 = vx
    # w3 = vy

    # || SAVED VALUES
    tList = np.arange(t0, tf, step=deltaT)
    w0List = [] # w0 = x
    w1List = [] # w1 = y
    w2List = [] # w0 = vx
    w3List = [] # w1 = vy

    w0_curr = w0_initial
    w1_curr = w1_initial
    w2_curr = w2_initial
    w3_curr = w3_initial

    for t in tList:

        if w1_curr < 0:
            break
        
        w0List.append(w0_curr)
        w1List.append(w1_curr)
        w2List.append(w2_curr)
        w3List.append(w3_curr)

        w0_prev = w0_curr
        w1_prev = w1_curr
        w2_prev = w2_curr
        w3_prev = w3_curr

        # k1
        k1_w0 = deltaT*f0(w2_prev, w3_prev, R, rho, C, m)
        k1_w1 = deltaT*f1(w2_prev, w3_prev, R, rho, C, m)
        k1_w2 = deltaT*f2(w2_prev, w3_prev, R, rho, C, m)
        k1_w3 = deltaT*f3(w2_prev, w3_prev, R, rho, C, m)

        # k2
        k2_w0 = deltaT*f0(w2_prev + 0.5 * k1_w2, w3_prev + 0.5 * k1_w3, R, rho, C, m)
        k2_w1 = deltaT*f1(w2_prev + 0.5 * k1_w2, w3_prev + 0.5 * k1_w3, R, rho, C, m)
        k2_w2 = deltaT*f2(w2_prev + 0.5 * k1_w2, w3_prev + 0.5 * k1_w3, R, rho, C, m)
        k2_w3 = deltaT*f3(w2_prev + 0.5 * k1_w2, w3_prev + 0.5 * k1_w3, R, rho, C, m)

        # k3
        k3_w0 = deltaT*f0(w2_prev + 0.5 * k2_w2, w3_prev + 0.5 * k2_w3, R, rho, C, m)
        k3_w1 = deltaT*f1(w2_prev + 0.5 * k2_w2, w3_prev + 0.5 * k2_w3, R, rho, C, m)
        k3_w2 = deltaT*f2(w2_prev + 0.5 * k2_w2, w3_prev + 0.5 * k2_w3, R, rho, C, m)
        k3_w3 = deltaT*f3(w2_prev + 0.5 * k2_w2, w3_prev + 0.5 * k2_w3, R, rho, C, m)

        # k4
        k4_w0 = deltaT*f0(w2_prev + k3_w2, w3_prev + k3_w3, R, rho, C, m)
        k4_w1 = deltaT*f1(w2_prev + k3_w2, w3_prev + k3_w3, R, rho, C, m)
        k4_w2 = deltaT*f2(w2_prev + k3_w2, w3_prev + k3_w3, R, rho, C, m)
        k4_w3 = deltaT*f3(w2_prev + k3_w2, w3_prev + k3_w3, R, rho, C, m)

        # w_n+1
        w0_curr = w0_prev + (1/6.)*k1_w0 + (1/3.)*k2_w0 + (1/3.)*k3_w0 +(1/6.)*k4_w0
        w1_curr = w1_prev + (1/6.)*k1_w1 + (1/3.)*k2_w1 + (1/3.)*k3_w1 +(1/6.)*k4_w1
        w2_curr = w2_prev + (1/6.)*k1_w2 + (1/3.)*k2_w2 + (1/3.)*k3_w2 +(1/6.)*k4_w2
        w3_curr = w3_prev + (1/6.)*k1_w3 + (1/3.)*k2_w3 + (1/3.)*k3_w3 +(1/6.)*k4_w3

    return tList, w0List, w1List, w2List, w3List



# || INITIAL CONDITIONS
v0 = 100
angle = math.radians(30)
w0_initial = 0
w1_initial = 0
w2_initial = v0 * np.cos(angle)
w3_initial = v0 * np.sin(angle)

tList, w0List, w1List, w2List, w3List = cannonBallDrag_plot(t0, tf, deltaT, R, rho, C, m, w0_initial, w1_initial, w2_initial, w3_initial)

plt.plot(w0List, w1List, label = f'({w0_initial:.2f}, {w1_initial:.2f}, {w2_initial:.2f}, {w3_initial:.2f})')
plt.tick_params(axis='both', labelsize=14)
plt.xlabel('x-position [m]', fontsize=14)
plt.ylabel('y-position [m]', fontsize=14)
plt.title(fr'Cannon Ball Trajectory, ($x_0$, $y_0$, $v^x_0$, $v^y_0$)', fontsize=14)
plt.legend(fontsize=14)
plt.show()



"""  
Part (c)
"""

mList = [0.1, 0.5, 1, 2, 5, 10, 50, 100, 200, 1000]
distanceTraveledList = []


for m in mList:
    tList, w0List, w1List, w2List, w3List = cannonBallDrag_plot(t0, tf, deltaT, R, rho, C, m, w0_initial, w1_initial, w2_initial, w3_initial)

    distanceTraveledList.append(w0List[-1])

    plt.plot(w0List, w1List, label = fr'$m = ${m}')
    
plt.tick_params(axis='both', labelsize=14)
plt.xlabel('x-position [m]', fontsize=14)
plt.ylabel('y-position [m]', fontsize=14)
plt.title(fr'Cannon Ball Trajectory, various masses', fontsize=14)
plt.legend(fontsize=14)
plt.show()


plt.plot(mList, distanceTraveledList)
plt.tick_params(axis='both', labelsize=14)
plt.xlabel('Mass, m [kg]', fontsize=14)
plt.ylabel('Total x-distance traveled, [m]', fontsize=14)
plt.title('Cannon Ball x-distance traveled vs mass of ball', fontsize=14)
plt.show()

plt.plot(mList[0: -2], distanceTraveledList[0:-2])
plt.tick_params(axis='both', labelsize=14)
plt.xlabel('Mass, m [kg]', fontsize=14)
plt.ylabel('Total x-distance traveled, [m]', fontsize=14)
plt.title('Cannon Ball x-distance traveled vs mass of ball', fontsize=14)
plt.show()