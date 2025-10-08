import casadi as ca
import numpy as np
from .support import *
def controller(x0_val,x0_init,theta_dot_ref,N,M):
    # --- Állapotok és bemenet ---
  
    # Csak egy legyen ne pedig N darab
    # Párhuzamos szimuláció és MPC párhuzamosan
    # Multiprocessing Process class és stb. de először dummy
    # Sebbességre valamilyen hard constraint (vagy soft attől függ a costban)
   
    w=[]
    lbw = []
    ubw = []
    J = 0
   
    dt = ca.MX.sym("dt")
    w   += [dt]
    lbw += [dt_min]   # paraméterekből
    ubw += [dt_max]
    theta = ca.MX.sym("theta")          # szög
    theta_dot = ca.MX.sym("theta_dot")  # szögsebesség
    h = ca.MX.sym("h")                  # magasság
    h_dot = ca.MX.sym("h_dot")          # magasságsebesség
    h_ddot = ca.MX.sym("h_ddot")        # döntési változók: u_k
    state = ca.vertcat(theta, theta_dot, h, h_dot)
    
    state_next, L_step = multiple_rk4_steps(state, h_ddot, dt)

    F = ca.Function("F", 
                    [state, h_ddot, dt],       # bemenetek
                    [state_next, L_step],      # kimenetek
                    ['state','h_ddot','dt'],   # bemenetnevek
                    ['state_next','L_step'])   # kimenetnevek
    g = ca.Function("g", [state[0],state[2]], [yoyo_height(state[0],state[2])])
   
    g=[]
    lbg = []
    ubg = []
    # "Lift" initial conditions
    state_k = ca.MX.sym('X0', 4)
    w += [state_k]

    lbw += [x0_val[0], x0_val[1],x0_val[2],x0_val[3]]
    ubw += [x0_val[0], x0_val[1],x0_val[2],x0_val[3]]



    target = ca.DM([0.0, float(theta_dot_ref), 1.0, 0.0])
    # Formulate the NLP
    for k in range(int(N/M)):
        # New NLP variable for the control
        h_ddot_k = ca.MX.sym('h_ddot_' + str(k))
        w   += [h_ddot_k]
        lbw += [U_min]
        ubw += [U_max]
        
       
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

        

     
    if k > int(N/M)*0.8:  
        target = ca.DM([0.0, float(theta_dot_ref), 1.0, 1.0])
        Q_terminal = ca.diag(ca.DM([1, 1, 100, 100]))
        err = state_k_end - target

        K_total = int(N/M)
        tau = (k+1) / float(K_total)     # normált idő 0..1 között
        alpha = 5.0                      # exponenciális erősség (hangolható)
        weight = ca.exp(alpha * tau)     # exponenciális növekedés

        J = J + weight * ca.mtimes([err.T, Q_terminal, err])




        '''
        Vajon az a jó hogy szigurúan előírjuk a drón sebességén kívűl az összes állapotot és hd_dotra írunk 
        egy soft contrait: Szerintem annyira nem, ha theta,thetha_dot, és h adott (hard constraittel) akkor
        h_dot-nak már kisebb halmazből kell kiekrülni, mennyire fontos hogy egy darab állaptot érjünk el?
        Nem elég ha egy állapothalmazba érünk el?
        Alábbiak lehetnek egy megoldás erre (nem 0 m/s ról indítjük a drónt)
        Ugye félő a szimulációban hogy az előző iterációbeli adatokkal dolgozunk, hogy 
        A másoidk iterációban beütjük a fejünket (mivel ott már picit magasabb mint 1 méterről van szó)
        Az biztos hogy nem műkődik ha h fix constriant és h_ddot nem (valamiért a constraitek betartása mellet bár optimális megoldást kapunk de h_dot nem lesz nulla, közel sem)
        Az a nagy baj, ha h==1 és h_dot=0 akkor kb azt írjuk elő hogy a drón egy ideig csak álljon egy helyben és úgy érjen le a jójó az ütközési pintra
        '''
    
       


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
        x0_init=w_opt
        visualize_results(w_opt)







    

    