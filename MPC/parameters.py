import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --- Paraméterek ---
L,m, r0, J, alpha, g, beta =0.6, 0.045, 0.005, 0.00003, 0.0000165, 9.81, 0.85


dt_max=0.05
dt_min=0.0001
#mint dt változó (periódis idő)
N = 100
M = 2
# Alsó és felső határ
# Állítsuk át (12 sok)
U_min = -7
U_max = 7
#
h_min=0.67
h_max=1.8
x0_val = np.array([0, 100, 1, 0])
x0_init = np.zeros([int(N/M),6]).flatten()  
x0_init = np.concatenate([x0_init, x0_val.flatten()])

theta_dot_ref=-117
