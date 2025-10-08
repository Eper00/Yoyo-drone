import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --- Paraméterek ---
L,m, r0, J, alpha, g, beta =0.6, 0.045, 0.005, 0.00003, 0.0000165, 9.81, 0.85


dt_max=0.05
dt_min=0.001
N =80
M = 1

U_min = -5.
U_max = 5.

h_min=0.7
h_max=1.8
x0_val = np.array([0, 100, 1, 0, dt_min])
x0_init = np.zeros([int(N/M),5]).flatten()  
x0_init = np.concatenate([x0_init, x0_val.flatten()])

theta_dot_ref=x0_val[1]/(-beta)
h_dot_ref=0
x_target=np.array([0,theta_dot_ref,1,1])
max_computing_time=0.5