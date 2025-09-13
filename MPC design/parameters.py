import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --- Paraméterek ---
L,m, r0, J, alpha, g, beta, dt =0.6, 0.045, 0.005, 0.00003, 0.0000165, 9.81, 0.85, 0.01
T=5
N = int(T/dt)
# Alsó és felső határ
U_min = 0.0
U_max = 10.0
x0_val = np.array([105.6, 0, 0.0, 0.0])