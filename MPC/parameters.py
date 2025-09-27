import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --- Paraméterek ---
L,m, r0, J, alpha, g, beta, dt =0.6, 0.045, 0.005, 0.00003, 0.0000165, 9.81, 0.85, 0.01
T=2.5
#mint dt változó (periódis idő)
N = int(T/dt)
M = 10
# Alsó és felső határ
# Állítsuk át (12 sok)
U_min = -10
U_max = 10
#
h_min=0.65
h_max=2
x0_val = np.array([0, 100, 1, 0])
h0=1
h0_dot=0
h0_ddot=0
theta_dot_ref=-117
