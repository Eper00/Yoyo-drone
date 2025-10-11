import casadi as ca
from acados_template import AcadosModel,AcadosOcp
from utils.parameters import m, r0, alpha, J, g, U_min, U_max ,x0_val ,h_min,h_max
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
    h_dot_dot = u

    f_expl = ca.vertcat(
        theta_dot,
        theta_ddot,
        h_dot,
        h_dot_dot
    )

    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_dot
    model.u = u
    return model
def formulate_ocp(Tf, N):
    ocp = AcadosOcp()
    model = create_model()
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()

    # horizon
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf

    Q_mat = 2*np.diag([1e3, 1e3, 1e-2, 1e-2])   # terminal
    R_mat = 2*np.diag([1e-2])                   # running (u^2)

    # --- PATH COST: LINEAR_LS formában: y = Vx * x + Vu * u ---
    ocp.cost.cost_type = 'LINEAR_LS'

    # ny = dim of (Vx*x + Vu*u) -> itt ny = nu = 1 (we only penalize u on path)
    ocp.cost.Vx = np.zeros((nu, nx))   # 1 x nx (no x in running cost)
    ocp.cost.Vu = np.eye(nu)           # 1 x 1 (u itself)
    ocp.cost.W = R_mat                 # 1 x 1

    ocp.cost.yref = np.zeros(nu)

    # --- TERMINAL COST (LINEAR_LS on states) ---
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.Vx_e = np.eye(nx)   # ny_e = nx
    ocp.cost.W_e = Q_mat
    ocp.cost.yref_e = np.zeros(nx)

    # ----- CONSTRAINTS -----
    ocp.constraints.lbu = np.array([U_min])
    ocp.constraints.ubu = np.array([U_max])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.x0 = x0_val[0:4]    # győződj meg, hogy x0_val hosszú-e
    ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])

    ocp.constraints.C = np.array([[0, 0, 1, 0]])
    ocp.constraints.lg = np.array([h_min])
    ocp.constraints.ug = np.array([h_max])

    # solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

    return ocp
