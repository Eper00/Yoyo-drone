from .model import create_model
from acados_template import AcadosOcp, AcadosOcpSolver
import numpy as np
from utils.parameters import *



def build_ocp_acados_variable(x0_val, x_target, N, Tf_guess, Tf_min, Tf_max, U_min, U_max, h_min, h_max):
    ocp = AcadosOcp()
    ocp.model = create_model()

    nx = ocp.model.x.size()[0]
    nu = 1

    ocp.dims.N = N
    ocp.solver_options.tf = 1.0  # normalizált időintervallum
    ocp.parameter_values = np.array([Tf_guess])  # kezdeti Tf érték

    # --- KÖLTSÉGFÜGGVÉNY ---
    # Running cost: csak u minimalizálása
    R = np.array([[1]])  # vezérlés súlya

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Csak u-t használjuk a futó költségben → nincs x komponens
    ocp.cost.W = R
    ocp.cost.Vx = np.zeros((nu, nx))  # nincs x az útvonal során
    ocp.cost.Vu = np.eye(nu)
    ocp.cost.yref = np.zeros(nu)  # referencia u = 0

    # Végső költség: csak állapot
    Q = np.diag([1, 1, 100, 100, 0])  # pl. t_norm ne legyen büntetve
    ocp.cost.W_e = Q
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref_e = np.hstack([x_target, 1.0])  # cél + t_norm = 1

    # --- KORLÁTOK ---
    ocp.constraints.lbx = np.array([0, -np.inf, h_min, -np.inf, 0])
    ocp.constraints.ubx = np.array([105, np.inf, h_max, np.inf, 1])
    ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4])
    ocp.constraints.x0 = np.array([*x0_val, 0.0])

    ocp.constraints.lbu = np.array([U_min])
    ocp.constraints.ubu = np.array([U_max])
    ocp.constraints.idxbu = np.array([0])

    # --- Tf mint döntési paraméter ---
    ocp.parameter_values = np.array([Tf_guess])
    ocp.constraints.lp = np.array([Tf_min])
    ocp.constraints.up = np.array([Tf_max])

    # Solver opciók
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"

    return ocp
