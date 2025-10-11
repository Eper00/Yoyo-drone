import numpy as np
import matplotlib.pyplot as plt

# --- jojó Paraméterek ---
L = 0.6
m_yoyo = 0.045
r_0 = 0.005
J_yoyo = 0.00003
alpha = 0.0000165
beta = 0.85   # reversal factor at x = 0
g = 9.81



# Szimuláció paraméterek
x0 = 105.6
y0 = 0
h0=1
T = 5
dt = 0.001
steps = int(T/dt)

# --- Függvények ---
def stepper_rk4(x, y, dt, m, r0, J, alpha, g, beta, h_ddot):
    r = lambda x: r0 + alpha * x
    I = lambda x: J + m * r(x)**2
    dy_dt = lambda x, y: -m * r(x) * (g + h_ddot) / I(x)

    k1x = dt * y
    k1y = dt * dy_dt(x, y)

    k2x = dt * (y + 0.5 * k1y)
    k2y = dt * dy_dt(x + 0.5 * k1x, y + 0.5 * k1y)

    k3x = dt * (y + 0.5 * k2y)
    k3y = dt * dy_dt(x + 0.5 * k2x, y + 0.5 * k2y)

    k4x = dt * (y + k3y)
    k4y = dt * dy_dt(x + k3x, y + k3y)

    x_next = x + (1/6) * (k1x + 2*k2x + 2*k3x + k4x)
    y_next = y + (1/6) * (k1y + 2*k2y + 2*k3y + k4y)

    if x > 0 and x_next <= 0:
        x_next = 0
        y_next = -beta * y

    return x_next, y_next

def yoyo_distance(theta, L, r0, alpha):
    return L - (r0 * theta + alpha * theta**2)
def update_drone_height(h,h_dot,h_ddot):
    h_dot+= dt*h_ddot
    h+= dt*h_dot
    
    return[h,h_dot]    



def trajectory(h0,x0, y0, dt, steps, m, r0, J, alpha, g, beta):
    x = x0
    y = y0
    h_dot=0
    h=h0
    traj = np.zeros((steps, 3))  # [x, y, h]
    traj[0, :] = [x, y,h0]

    for i in range(1, steps):
        if y<0:
            h_ddot=1.5
        else:
            h_ddot=-1.5
        x_new, y_new = stepper_rk4(x, y, dt, m, r0, J, alpha, g, beta, h_ddot)
        h,h_dot=update_drone_height(h,h_dot,h_ddot)
        traj[i, :] = [x_new, y_new, h]
        x, y = x_new, y_new
    return traj

# --- Fázistér (quiver) előkészítése ---
X, Y = np.meshgrid(np.arange(0, 301, 10), np.arange(-300, 301, 10))
U = np.zeros_like(X, dtype=float)
V = np.zeros_like(Y, dtype=float)

idx_pos = X > 0
R = r_0 + alpha * X[idx_pos]
U[idx_pos] = Y[idx_pos]
V[idx_pos] = -(m_yoyo * R / (J_yoyo + m_yoyo * R**2)) * (g)

# --- Trajektória ---
traj = trajectory(h0,x0, y0, dt, steps, m_yoyo, r_0, J_yoyo, alpha, g, beta)
x_traj, y_traj ,h= traj.T

# --- Kirajzolás ---
plt.figure(figsize=(10, 6))
plt.quiver(X, Y, U, V, color='lightgrey', alpha=0.9)
plt.plot(x_traj, y_traj, 'r', linewidth=2, label='Trajektória')
plt.xlabel('x [rad]')
plt.ylabel('dx/dt [rad/s]')
plt.title('Yoyo fázistér diagram')
plt.legend()
plt.grid(True)
plt.show()