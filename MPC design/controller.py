import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from parameters import *
from constraints import rk4_step, yoyo_height

class Control:
    def __init__(self, N, dt,x0_val):

        self.N = N
        self.dt = dt
        self.U = ca.MX.sym('U', self.N)  

        self.nx = 4  # [theta, theta_dot, h,h_dot]
        self.nu = 1  # control input
        
        # Bounds
        self.u_min = U_min
        self.u_max = U_max
        self.constraints=[]
        
        # Build NLP solver
        self._build_nlp(x0_val)
    
        
    def _build_nlp(self, x0_val):
        # Döntési változók: csak a kontrollok
        
        # Szimbolikus kezdőállapot (paraméterként)
        xk = ca.MX(x0_val)  # kezdőállapot
        X = [xk]            # állapotok listája

        constraints = []

        for k in range(self.N):
            uk = self.U[k]

            # --- Dinamika ---
            x_next = rk4_step(xk, uk)          # RK4 lépés
            constraints.append(x_next - x_next)  # ideális: 0, ha equality, vagy
            
           

            # Következő állapot
            xk = x_next
            X.append(xk)

        # --- Konvertálás mátrixba ---
        X = ca.hcat(X)
        g = ca.vertcat(*constraints)

        # --- Flatten döntési változók ---
        w = ca.reshape(self.U , -1, 1)
        cost = ca.sumsqr(self.U )

        # --- NLP definíció ---
        nlp = {'x': w, 'f': cost, 'g': g}

        # Solver opciók
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # --- Korlátok numerikusan ---
        # Dinamika egyenlőség: lbg = ubg = 0
        ng = g.shape[0]  # constraints száma
        lbg = np.zeros((ng, 1))
        ubg = np.zeros((ng, 1))
        


        return X, g, lbg, ubg



    def compute_control(self, x0):
        # Initial guess
        w0 = ca.MX.zeros((self.nu * self.N, 1))  # numerikus inicializálás

        # Constraint bounds
        ng = 4 * self.N  # két állapot és két vezérlő korlát per lépés
        lb_g = np.zeros((ng, 1))
        ub_g = np.zeros((ng, 1))

        # A solver hívásakor a numerikus x0-t adjuk át paraméterként
        sol = self.solver(x0=w0, lbg=lb_g, ubg=ub_g, p=x0)  
        w_opt = sol['x'].full().flatten()

        # Első kontroll jel
        u_opt = w_opt[0]
        return u_opt

P=Control(N,dt,x0_val)
print(P.constraints[0])