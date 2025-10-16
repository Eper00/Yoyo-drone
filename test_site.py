from utils.parameters import *
from utils.support import multiple_rk4_steps
from MPC_Acados.model import control_loop_acados
from MPC_Casadi.control import Controller
from acados_template import AcadosModel,AcadosOcp,AcadosOcpSolver
#control_loop_acados(2,x0_val)
C=Controller(x0_val, x0_init, N, 
                                M, params,multiple_rk4_steps)
C.control_loop(3)
