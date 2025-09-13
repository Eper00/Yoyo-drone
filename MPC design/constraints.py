from parameters import *
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
    k4y = dt * dy_dt(x + k3x, y + k3y)

    x_next = x + (1/6) * (k1x + 2*k2x + 2*k3x + k4x)
    y_next = y + (1/6) * (k1y + 2*k2y + 2*k3y + k4y)

    # Magasság update
    h_dot_next = h_dot + dt*u
    h_next = h + dt*h_dot

    # visszapattanás (reflexió) ha x átlép 0-n
    x_next = ca.if_else(ca.logic_and(x>0, x_next<=0), 0, x_next)
    y_next = ca.if_else(ca.logic_and(x>0, x_next<=0), -beta*y, y_next)

    return ca.vertcat(x_next, y_next, h_next, h_dot_next)



def yoyo_height (theta,h):
    return h-L+r0*theta+(alpha/2)*theta**2