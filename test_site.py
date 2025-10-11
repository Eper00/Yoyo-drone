from MPC_Casadi.control import controller,control_loop
from utils.support import *
from MPC_Acados.control import build_ocp_acados_variable
from MPC_Acados.model import create_model,formulate_ocp
from acados_template import AcadosModel,AcadosOcp,AcadosOcpSolver
N_horizon = 20
Tf = 1.0
ocp = formulate_ocp(Tf, N_horizon)



    ## solve using acados
    # create acados solver
ocp_solver = AcadosOcpSolver(ocp,verbose=False)
    # solve with acados
status = ocp_solver.solve()
if status != 0:
    raise Exception(f'acados returned status {status}.')
    # get solution
result = ocp_solver.store_iterate_to_obj()




