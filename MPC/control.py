import casadi as ca
import numpy as np
from support import *
def controller(x0_val,x0_init,theta_dot_ref,N,M):
    # --- Állapotok és bemenet ---
    theta = ca.MX.sym("theta")          # szög
    theta_dot = ca.MX.sym("theta_dot")  # szögsebesség
    h = ca.MX.sym("h")                  # magasság
    h_dot = ca.MX.sym("h_dot")          # magasságsebesség
    h_ddot = ca.MX.sym("h_ddot")        # döntési változók: u_k


    dt = ca.MX.sym("dt")

    state = ca.vertcat(theta, theta_dot, h, h_dot)
    L=h_ddot**2

    state_next, L_step = multiple_rk4_steps(state, h_ddot, dt)

    F = ca.Function("F", 
                    [state, h_ddot, dt],       # bemenetek
                    [state_next, L_step],      # kimenetek
                    ['state','h_ddot','dt'],   # bemenetnevek
                    ['state_next','L_step'])   # kimenetnevek
    g = ca.Function("g", [state[0],state[2]], [yoyo_height(state[0],state[2])])
    w=[]
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []
    # "Lift" initial conditions
    state_k = ca.MX.sym('X0', 4)
    w += [state_k]

    lbw += [x0_val[0], x0_val[1],x0_val[2],x0_val[3]]
    ubw += [x0_val[0], x0_val[1],x0_val[2],x0_val[3]]

    # Formulate the NLP
    for k in range(int(N/M)):
        # New NLP variable for the control
        h_ddot_k = ca.MX.sym('h_ddot_' + str(k))
        w   += [h_ddot_k]
        lbw += [U_min]
        ubw += [U_max]
        # Új döntési változó: dt
        dt = ca.MX.sym("dt")
        w   += [dt]
        lbw += [dt_min]   # paraméterekből
        ubw += [dt_max]





        # Integrate till the end of the interval
        Fk = F(state_k, h_ddot_k,dt)
        state_k_end = Fk[0]
        J=J+Fk[1]
        # New NLP variable for state at end of interval
        state_k = ca.MX.sym('state_' + str(k+1), 4)
        w   += [state_k]
        lbw += [0, -1*ca.inf,h_min,-1*ca.inf]
        ubw += [105,ca.inf,h_max,ca.inf]
        # Add equality constraint
        g   += [state_k_end-state_k]
        lbg += [0, 0,0,0]
        ubg += [0, 0,0,0]
    
        
        if k == int(N/M) - 1:
          
            g   += [state_k_end[0] - 0]
            lbg += [0]
            ubg += [0]
            g   += [state_k_end[2] - 1]             # h(T) = 1
            lbg += [0]
            ubg += [0]

            g   += [state_k_end[1] - theta_dot_ref]  # theta_dot(T) = theta_dot_ref
            lbg += [0]
            ubg += [0]
            


    
    
                    





    # Create an NLP solver
    prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
    solver = ca.nlpsol('solver', 'ipopt', prob)

    # Solve the NLP
    sol = solver(x0=x0_init, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()

    # Plot the solution
     
    return w_opt


def control_loop(cyclenumber,x0_val,x0_init,theta_dot_ref,N,M):
    for _ in range(cyclenumber): 
        w_opt=controller(x0_val,x0_init,theta_dot_ref,N,M)
        x0_val[0]=w_opt[-4]
        x0_val[1]=w_opt[-3]*-beta
        x0_val[2]=w_opt[-2]
        x0_val[3]=w_opt[-1]
        visualize_results(w_opt)
        x0_init=w_opt








    

    