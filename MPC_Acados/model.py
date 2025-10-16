import casadi as ca
from acados_template import AcadosModel,AcadosOcp,AcadosOcpSolver
from utils.parameters import *
from utils.support import get_x_u_traj,visualize_results_acados
import numpy as np



def create_model():
    model = AcadosModel()
    model.name = "yoyo_model"

    # Állapotok
    theta     = ca.SX.sym("theta")
    theta_dot = ca.SX.sym("theta_dot")
    h         = ca.SX.sym("h")
   


    h_dot     = ca.SX.sym("h_dot")
    theta_ddot = ca.SX.sym("theta_ddot")
    h_ddot = ca.SX.sym("h_ddot")
    u = h_ddot



    x = ca.vertcat(theta,theta_dot,h,h_dot)
    x_dot=ca.vertcat(theta_dot,theta_ddot, h_dot,h_ddot)
    # Dinamika
    theta_ddot = -m * (r0 + alpha * theta) * (g + u) / (J + m * (r0 + alpha * theta)**2)
    
    dt=ca.SX.sym("dt")
    f_expl = dt*ca.vertcat(
        theta_dot,
        theta_ddot,
        h_dot,
        u,
    )

   
    model.x = x
    model.xdot = x_dot
    model.f_expl_expr = f_expl
    model.u = ca.vertcat(u,dt)
    return model

def formulate_ocp():
    ocp = AcadosOcp()
    model = create_model()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    # horizon
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = 1

    Q_mat = np.diag([1, 1, 100, 100])   # terminal
    R_mat = np.diag([1,400])                 # running (u^2)

    # --- PATH COST: LINEAR_LS formában: y = Vx * x + Vu * u ---
    ocp.cost.cost_type = 'LINEAR_LS'

    # ny = dim of (Vx*x + Vu*u) -> itt ny = nu = 1 (we only penalize u on path)
    ocp.cost.Vx = np.zeros((nu, nx))   # 1 x nx (no x in running cost)
    ocp.cost.Vu = np.eye(nu)           # 1 x 1 (u itself)
    ocp.cost.W = R_mat                 # 1 x 1

    ocp.cost.yref = np.zeros(nu)

    # --- TERMINAL COST (LINEAR_LS on states) ---
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.W_e = Q_mat
    ocp.cost.yref_e = x_target

    # ----- CONSTRAINTS -----
    ocp.constraints.lbu = np.array([U_min,0.1])
    ocp.constraints.ubu = np.array([U_max,3])
    ocp.constraints.idxbu = np.array([0,1])

    ocp.constraints.x0 = x0_val[0:-1] 
    ocp.constraints.Jbx = np.array([
        [1, 0, 0, 0,0],  
        [0, 0, 1, 0,0],
    ])
    ocp.constraints.lbx = np.array([0, h_min])
    ocp.constraints.ubx = np.array([105, h_max])


    # solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

    return ocp
def control_loop_acados(cycle_times,x0_val):
    x0 = x0_val[0:-1]

    # egyszer hozzuk létre az OCP-t és a solvert
    ocp = formulate_ocp()
    ocp_solver = AcadosOcpSolver(ocp, verbose=False)

    for _ in range(cycle_times):
        # frissítsd az aktuális kezdőállapotot
        ocp_solver.set(0, "lbx", x0)
        ocp_solver.set(0, "ubx", x0)

        # futtasd a solvert
        status = ocp_solver.solve()
        if status != 0:
            raise Exception(f'acados returned status {status}.')

        simX, simU = get_x_u_traj(ocp_solver, N)

        x0_val = np.array([simX[-1,0], simX[-1,1]*-beta, simX[-1,2], simX[-1,3]])

        # vizualizáció
        visualize_results_acados(simX, simU, N+1, 1)
        print(simU[-1,-1])





