import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from stepper_functions import *
import csv

# --- Állapotok és bemenet ---
theta = ca.MX.sym("theta")          # szög
theta_dot = ca.MX.sym("theta_dot")  # szögsebesség
h = ca.MX.sym("h")                  # magasság
h_dot = ca.MX.sym("h_dot")          # magasságsebesség
h_ddot = ca.MX.sym("h_ddot")        # döntési változók: u_k

state = ca.vertcat(theta, theta_dot, h, h_dot)
L=h_ddot**2

state_next, L_step = multiple_rk4_steps(state, h_ddot)
F = ca.Function("F", [state, h_ddot], [state_next, L_step], ['state','u'], ['state_next','L_step'])
g = ca.Function("g", [state[0],state[2]], [yoyo_height(state[0],state[2])])
w=[]
w0 = []
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
w0 += [x0_val[0], x0_val[1],x0_val[2],x0_val[3]]

# Formulate the NLP
for k in range(int(N/M)):
    # New NLP variable for the control
    h_ddot_k = ca.MX.sym('h_ddot_' + str(k))
    w   += [h_ddot_k]
    lbw += [U_min]
    ubw += [U_max]
    w0  += [0]

    # Integrate till the end of the interval
    Fk = F(state_k, h_ddot_k)
    state_k_end = Fk[0]
    J=J+Fk[1]
    # New NLP variable for state at end of interval
    state_k = ca.MX.sym('state_' + str(k+1), 4)
    w   += [state_k]
    lbw += [0, -1*ca.inf,h_min,-1*ca.inf]
    ubw += [105,ca.inf,h_max,ca.inf]
    w0  += [0, 0,0,0]
    # Add equality constraint
    g   += [state_k_end-state_k]
    lbg += [0, 0,0,0]
    ubg += [0, 0,0,0]
    # Add midpoint constraint
    '''
    if k == N//2 - 1:   # T/2 időpillanat
        g   += [state_k_end[0] - theta_ref]  # theta(T/2) = theta_ref
        lbg += [0]
        ubg += [0]
    '''
    # Add final constraint
    
    if k == int(N/M) - 1:
        
        g   += [state_k_end[0] - 0]  # theta_dot(T) = theta_dot_ref
        lbg += [0]
        ubg += [0]
        g   += [state_k_end[1] - theta_dot_ref]  # theta_dot(T) = theta_dot_ref
        lbg += [0]
        ubg += [0]
        g   += [state_k_end[2] - 1]             # h(T) = 1
        lbg += [0]
        ubg += [0]

  
  
                





    # Create an NLP solver
prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
solver = ca.nlpsol('solver', 'ipopt', prob)

# Solve the NLP
# inicilaizéció
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()

# Plot the solution
theta_opt = w_opt[0::5]
theta_dot_opt = w_opt[1::5]
h_opt = w_opt[2::5]
h_dot_opt = w_opt[3::5]
h_ddot_opt = w_opt[4::5]
tgrid = [M*T/N*k for k in range(int(N/M)+1)]


import matplotlib.pyplot as plt

plt.figure(1)
plt.clf()
plt.plot(tgrid, theta_opt, '--')
plt.plot(tgrid, theta_dot_opt, '--')
plt.grid()
plt.show()
plt.figure(1)
plt.clf()
plt.plot(tgrid, h_opt, '--')
plt.plot(tgrid, yoyo_height(theta_opt,h_opt), '--')

plt.grid()
plt.show()
tgrid = [M*T/N*k for k in range(int(N/M)+1)]
plt.figure(1)
plt.clf()
plt.plot(tgrid, h_dot_opt, '--')



plt.grid()
plt.show()
csvdata=[]
head = ["h_opt", "h_dot_opt", "h_ddot_opt"]

csvdata = list(zip(h_opt, h_dot_opt, h_ddot_opt))

with open('MPC_input.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(head)      
    writer.writerows(csvdata)  



    

    