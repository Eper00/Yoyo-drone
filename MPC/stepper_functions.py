from parameters import *
# --- Dinamika: RK4 integrátor ---
def rk4_step(state, u):
    theta     = state[0]
    theta_dot = state[1]
    h         = state[2]
    h_dot     = state[3]
    r = lambda xx: r0 + alpha * xx
    I = lambda xx: J + m * r(xx)**2
    dy_dt = lambda xx, yy: -m * r(xx) * (g + u) / I(xx)

    k1x = dt * theta_dot
    k1y = dt * dy_dt(theta, theta_dot)

    k2x = dt * (theta_dot + 0.5 * k1y)
    k2y = dt * dy_dt(theta + 0.5 * k1x, theta_dot + 0.5 * k1y)

    k3x = dt * (theta_dot + 0.5 * k2y)
    k3y = dt * dy_dt(theta + 0.5 * k2x, theta_dot + 0.5 * k2y)

    k4x = dt * (theta_dot + k3y)
    k4y = dy_dt(theta + k3x, theta_dot + k3y) * dt

    x_next = theta + (1/6) * (k1x + 2*k2x + 2*k3x + k4x)
    y_next = theta_dot + (1/6) * (k1y + 2*k2y + 2*k3y + k4y)




    k1_h = dt * h_dot
    k1_hdot = dt * u

    k2_h = dt * (h_dot + 0.5*k1_hdot)
    k2_hdot = dt * u  # u konstans

    k3_h = dt * (h_dot + 0.5*k2_hdot)
    k3_hdot = dt * u

    k4_h = dt * (h_dot + k3_hdot)
    k4_hdot = dt * u
    # Magasság update
    h_next = h + (1/6)*(k1_h + 2*k2_h + 2*k3_h + k4_h)
    h_dot_next = h_dot + (1/6)*(k1_hdot + 2*k2_hdot + 2*k3_hdot + k4_hdot)
    L_step = u**2
    state_next=state_next = ca.vertcat(x_next, y_next, h_next, h_dot_next)
    return state_next, L_step


def multiple_rk4_steps(state, h_ddot):
    state_k = state
    Q = 0
    for _ in range(M):
        state_k, L_step = rk4_step(state_k, h0_ddot)
        Q = Q + L_step*dt  # integrált költség = ∫u^2 dt
    return state_k, Q
yoyo_height = lambda theta, h: h - L + r0*theta + (alpha/2)*theta**2