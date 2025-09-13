import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --- Paraméterek ---
m, r0, J, alpha, g, beta, dt = 0.045, 0.005, 0.00003, 0.0000165, 9.81, 0.85, 0.01
N = 500

# --- Állapotok és bemenet ---
x = ca.MX.sym("x")        # szög
y = ca.MX.sym("y")        # szögsebesség
h = ca.MX.sym("h")        # magasság
h_dot = ca.MX.sym("h_dot") # magasságsebesség
U = ca.MX.sym("u", N)        # döntési változók: u_k

# --- Kezdő állapot ---
x0_val = np.array([105.6, 0.0, 0.0, 0.0])
state = ca.vertcat(x, y, h, h_dot)

# --- Dinamika: RK4 integrátor ---
def rk4_step(state, u):
    x, y, h, h_dot = state[0], state[1], state[2], state[3]

    r = lambda xx: r0 + alpha * xx
    I = lambda xx: J + m * r(xx)**2
    dy_dt = lambda xx, yy: -m * r(xx) * (g + u) / I(xx)

    k1x = dt * y
    k1y = dt * dy_dt(x, y)

    k2x = dt * (y + 0.5 * k1y)
    k2y = dt * dy_dt(x + 0.5 * k1x, y + 0.5 * k1y)

    k3x = dt * (y + 0.5 * k2y)
    k3y = dt * dy_dt(x + 0.5 * k2x, y + 0.5 * k2y)

    k4x = dt * (y + k3y)
    k4y = dy_dt(x + k3x, y + k3y) * dt

    x_next = x + (1/6) * (k1x + 2*k2x + 2*k3x + k4x)
    y_next = y + (1/6) * (k1y + 2*k2y + 2*k3y + k4y)

    # Magasság update
    h_dot_next = h_dot + dt * u
    h_next = h + dt * h_dot

    # Visszapattanás (reflexió) ha x átlép 0-n
    x_next = ca.if_else(ca.logic_and(x > 0, x_next <= 0), 0, x_next)
    y_next = ca.if_else(ca.logic_and(x > 0, x_next <= 0), -beta * y, y_next)

    return ca.vertcat(x_next, y_next, h_next, h_dot_next)

# CasADi függvény
u_sym = ca.MX.sym("u")  # egyetlen döntési változó
f = ca.Function("f", [state, u_sym], [rk4_step(state, u_sym)])

# --- Trajektória a döntési változók függvényében ---
X = [ca.MX(x0_val)]
xk = ca.MX(x0_val)
for k in range(N):
    xk = f(xk, U[k])
    X.append(xk)

X = ca.hcat(X)

# --- Korlátok ---
constraints = [X[:, 0] - x0_val]  # kezdőállapot fix
lbg = np.zeros(4)
ubg = np.zeros(4)

g = ca.vertcat(*constraints)

# --- Célfüggvény: minimalizáljuk az energiafelhasználást ---
cost = ca.sumsqr(U)

nlp = {'x': U, 'f': cost, 'g': g}

# Solver
solver = ca.nlpsol('solver', 'ipopt', nlp)
sol = solver(x0=np.zeros(N), lbg=lbg, ubg=ubg)
U_opt = sol['x'].full().flatten()  # Nx1 -> N

# --- Kezdő állapot ---
xk = x0_val.copy()
X_traj = [xk]

for k in range(N):
    xk = f(xk, U_opt[k])
    X_traj.append(xk.full().flatten())

X_traj = np.array(X_traj)  # (N+1, 4)
plt.figure()
plt.plot(X_traj[:,2])  # magasság
plt.plot(X_traj[:,0])  # szög
plt.xlabel("lépések")
plt.ylabel("érték")
plt.legend(["h", "x"])
plt.show()