from MPC.control import controller,control_loop
from MPC.support import *


w_opt=control_loop(3,x0_val,x0_init,theta_dot_ref,N,M)

#simualtion_verdict(x0_val,2.5,0.001,0.5)

