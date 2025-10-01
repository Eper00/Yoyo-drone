from parameters import *
import csv

# --- Dinamika: RK4 integrátor ---
def rk4_step(state, u,dt):
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


def multiple_rk4_steps(state, h_ddot,dt):
    state_k = state
    Q = 0
    for _ in range(M):
        state_k, L_step = rk4_step(state_k, h_ddot,dt)
        Q = Q + L_step*dt  # integrált költség = ∫u^2 dt
    return state_k, Q

def simulate(state_init, T, dt):
    states = []
    state_k = state_init

    # első elem: csak state_init (4 elem)
    states.append(np.array(state_k).flatten())

    # minden lépés: state_k + 0 + dt (6 elem)
    
    state_k, _ = rk4_step(state_k, 0, dt)
    vec = ca.vertcat( 0, dt,state_k)
    states.append(np.array(vec).flatten())

    # végül egyetlen sorvektor
    states_flat = np.concatenate(states)
    return states_flat

yoyo_height = lambda theta, h: h - L + r0*theta + (alpha/2)*theta**2
def disassemble(opt):
    theta_opt = opt[0::6]
    theta_dot_opt = opt[1::6]
    h_opt = opt[2::6]
    h_dot_opt = opt[3::6]
    h_ddot_opt = opt[4::6]
    dt_opt = opt[5::6]
    return [theta_opt,theta_dot_opt,h_opt,h_dot_opt,h_ddot_opt,dt_opt]
def simualtion_verdict(state_init, T, dt_1,dt_2):

    states_1=simulate(state_init,T,dt_1)
    states_2=simulate(state_init,T,dt_2)
    [theta_opt_1,theta_dot_opt_1,h_opt_1,h_dot_opt_1,h_ddot_opt_1,dt_opt_1]=disassemble(states_1)
    [theta_opt_2,theta_dot_opt_2,h_opt_2,h_dot_opt_2,h_ddot_opt_2,dt_opt_2]=disassemble(states_2)
    tgrid_1 = [0.0]
    for k in range(len(dt_opt_1)):
        tgrid_1.append(tgrid_1[-1] + dt_opt_1[k])
    tgrid_2 = [0.0]
    for k in range(len(dt_opt_2)):
        tgrid_2.append(tgrid_2[-1] + dt_opt_2[k])
    plt.figure()
    plt.plot(tgrid_1, theta_opt_1, '-', label="theta with dt_1={}".format(dt_1))
    plt.plot(tgrid_2, theta_opt_2, '-', label="theta with dt_2={}".format(dt_2))

    plt.grid()
    plt.legend()
    plt.show()




def visualize_results(w_opt):
    [theta_opt,theta_dot_opt,h_opt,h_dot_opt,h_ddot_opt,dt_opt]=disassemble(w_opt)

    tgrid = [0.0]
    for k in range(len(dt_opt)):
        tgrid.append(tgrid[-1] + dt_opt[k])
    plt.figure()
    plt.plot(tgrid, theta_opt, '--', label="theta")
    plt.plot(tgrid, theta_dot_opt, '--', label="theta_dot")
    plt.grid()
    plt.legend()
    plt.show()

    # h és a yoyo_height
    plt.figure()
    plt.plot(tgrid, h_opt, '--', label="h")
    plt.plot(tgrid, yoyo_height(theta_opt, h_opt), '--', label="yoyo_height")
    plt.grid()
    plt.legend()
    plt.show()

    # h_dot
    plt.figure()
    plt.plot(tgrid, h_dot_opt, '--', label="h_dot")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.step(tgrid[:-1], h_ddot_opt, where="post", label="h_ddot")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(dt_opt, label="dt")
    plt.grid()
    plt.legend()
    plt.show()
def save_data(w_opt):
    h_opt = w_opt[2::6]
    h_dot_opt = w_opt[3::6]
    h_ddot_opt = w_opt[4::6]
    csvdata=[]
    head = ["h_opt", "h_dot_opt", "h_ddot_opt"]

    csvdata = list(zip(h_opt, h_dot_opt, h_ddot_opt))

    with open('MPC_input.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)      
        writer.writerows(csvdata)  