from control import controller,control_loop
from parameters import *
from support import save_data,visualize_results,simulate,simualtion_verdict


w_opt=control_loop(3,x0_val,x0_init,theta_dot_ref,N,M)
#simualtion_verdict(x0_val,2.5,0.001,0.05)

