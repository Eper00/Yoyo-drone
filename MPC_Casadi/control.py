import casadi as ca
from utils.support import visualize_results_casadi
class Controller:
    def __init__(self, x0_val, x0_init, N, M, params,dynamic_function):
        """
        Controller osztály inicializálása.
        """
        self.x0_val = x0_val
        self.x0_init = x0_init
        self.N = N
        self.M = M
        
        # Külső paraméterek (pl. határok, célállapot, stb.)
        self.dt_min = params["dt_min"]
        self.dt_max = params["dt_max"]
        self.U_min = params["U_min"]
        self.U_max = params["U_max"]
        self.h_min = params["h_min"]
        self.h_max = params["h_max"]
        self.x_target = params["x_target"]
        self.beta = params["beta"]
        self.multiple_rk4_steps=dynamic_function


    def build_nlp(self):

        w = []
        lbw = []
        ubw = []
        J = 0

        # Diszkrét lépésidő döntési változó
        dt = ca.MX.sym("dt")
        w += [dt]
        lbw += [self.dt_min]
        ubw += [self.dt_max]

        # Állapotváltozók
        theta = ca.MX.sym("theta")
        theta_dot = ca.MX.sym("theta_dot")
        h = ca.MX.sym("h")
        h_dot = ca.MX.sym("h_dot")
        h_ddot = ca.MX.sym("h_ddot")  # vezérlés

        state = ca.vertcat(theta, theta_dot, h, h_dot)
        state_next, L_step = self.multiple_rk4_steps(state, h_ddot, dt)

        F = ca.Function("F", 
                        [state, h_ddot, dt],
                        [state_next, L_step],
                        ['state', 'h_ddot', 'dt'],
                        ['state_next', 'L_step'])

        g = []
        lbg = []
        ubg = []

        # Kezdeti állapot
        state_k = ca.MX.sym('X0', 4)
        w += [state_k]
        lbw += list(self.x0_val[0:-1])
        ubw += list(self.x0_val[0:-1])

        # Fő ciklus
        for k in range(int(self.N / self.M)):
            h_ddot_k = ca.MX.sym(f'h_ddot_{k}')
            w += [h_ddot_k]
            lbw += [self.U_min]
            ubw += [self.U_max]

            Fk = F(state_k, h_ddot_k, dt)
            state_k_end = Fk[0]
            J += Fk[1]

            # Új állapot
            state_k = ca.MX.sym(f'state_{k+1}', 4)
            w += [state_k]
            lbw += [0, -ca.inf, self.h_min, -ca.inf]
            ubw += [105, ca.inf, self.h_max, ca.inf]

            # Dinamikai egyenlet constraint
            g += [state_k_end - state_k]
            lbg += [0, 0, 0, 0]
            ubg += [0, 0, 0, 0]

            # Terminális költség utolsó lépésben
            if k == int(self.N / self.M) - 1:
                Q_terminal = ca.diag(ca.DM([1, 10, 1000, 100]))    
                err = state_k_end - self.x_target

                J = J + 150*ca.mtimes([err.T, Q_terminal, err])

        prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
        return prob, w, lbw, ubw, lbg, ubg

    def solve(self):

        prob, w, lbw, ubw, lbg, ubg = self.build_nlp()
        solver = ca.nlpsol('solver', 'ipopt', prob)
        sol = solver(x0=self.x0_init, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()
        
      
        return w_opt

    def control_loop(self, cyclenumber):

        for _ in range(cyclenumber):
            print(self.x0_val)
            w_opt = self.solve()
            self.x0_val[0] = w_opt[-4]
            self.x0_val[1] = w_opt[-3] * -self.beta
            self.x0_val[2] = w_opt[-2]
            self.x0_val[3] = w_opt[-1]
            self.x0_val[4] = w_opt[0]
            self.x0_init = w_opt
            # Eredmények kirajzolása
            visualize_results_casadi(w_opt)


