import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --- Paraméterek ---
L,m, r0, J, alpha, g, beta, dt =0.6, 0.045, 0.005, 0.00003, 0.0000165, 9.81, 0.85, 0.01
T=2.7
N = int(T/dt)
M = 4 # RK4 steps per interval

# Alsó és felső határ
U_min = -5
U_max = 5
x0_val = np.array([0, 126.4, 1, 0])
h0=1
h0_dot=0
h0_ddot=0
d_ref=0.5
theta_dot_ref=1/beta*ca.sqrt((2*m*g*d_ref)/(m*r0**2+J))