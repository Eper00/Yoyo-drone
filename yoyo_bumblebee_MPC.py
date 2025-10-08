"""
Module that contains bumblebees with yoyos attached.
"""

from typing import Optional
from xml.etree import ElementTree as ET
import mujoco
import numpy as np
import csv
import pandas as pd
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone import bumblebee
from multiprocessing import Process,Queue
from scripts.my_code.MPC import control, parameters,support
class YoyoBumblebee1DOF_MPC(bumblebee.Bumblebee):
    """
    Class that extends the Bumblebee to have a 1-DOF yoyo on it.
    """

    L = parameters.L
    m_yoyo = parameters.m
    r_0 = parameters.r0
    J_yoyo = parameters.J
    alpha = parameters.alpha
    beta = parameters.beta
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



        # --- kezdeti állapotértékek ---
        self.theta = parameters.x0_val[0]
        self.theta_dot = parameters.x0_val[1]
        self.h = parameters.x0_val[2]
        self.h_dot = parameters.x0_val[3]
        self.h_ddot = 0
        self.theta_ddot = 0
        self.d = 0
        self.r = 0

        # --- kiegészítő állapotváltozók ---
        self.T = 0
        self.z = 0
        self.z_dot = 0
        self.z_ddot = 0
        self.T_imp = 0

        # --- MPC eredmények ---
        self.h_opt = []
        self.h_dot_opt = []
        self.h_ddot_opt = []
        self.dt_opt = 0.001

        # --- interpolált pályák ---
        self.h_opt_interp = []
        self.h_dot_opt_interp = []
        self.h_ddot_opt_interp = []

        # --- korábbi MPC eredmények ---
        self.h_opt_interp_before = []
        self.h_dot_opt_interp_before = []
        self.h_ddot_opt_interp_before = []

        # --- időzítés és vezérlés ---
        self.dt = 0.001
        self.k = 0
        self.index_write = 0
        self.csvdata = []
        self.time = 0
        self.max_computing_time = parameters.max_computing_time
        self.index_control = 0
        self.switch = 0

        # --- párhuzamosítás ---
        self.process = None
        self.queue = None
    @classmethod
    def get_identifiers(cls) -> Optional[list[str]]:
        return ["YoyoBumblebee1DOF_PD"]

    def create_xml_element(self, pos: str, quat: str, color: str) -> dict[str, list[ET.Element]]:

        
        # grab the parent's xml to augment with our own
        bumblebee_xml = super().create_xml_element(pos, quat, color)
        drone_body = bumblebee_xml["worldbody"][0]
        actuators = bumblebee_xml["actuator"]
        sensors = bumblebee_xml["sensor"]
        yoyo_name = f"{self.name}_yoyo"
        site_name = self.name + "_yoyo_base"

        yoyo = ET.SubElement(drone_body, "body", name=yoyo_name, pos="0.0085 0 0")

        ET.SubElement(yoyo, "site", name=site_name, pos=f"0 0 0", type="sphere", size="0.003")

        ET.SubElement(yoyo, "joint", name=f"{yoyo_name}_joint", type="slide")
        
        yoyo_head = ET.SubElement(yoyo, "body", name=f"{yoyo_name}_head", pos=f"0 0 0", euler="1.5708 0 0")

        ET.SubElement(yoyo_head, "geom", type="cylinder", pos="0 0 0", euler="0 0 0", size="0.005 0.005", mass="0.0001")
        ET.SubElement(yoyo_head, "geom", type="cylinder", pos="0 0 0.015", euler="0 0 0", size="0.050 0.010", mass="0.0001")
        ET.SubElement(yoyo_head, "geom", type="cylinder", pos="0 0 -0.015", euler="0 0 0", size="0.050 0.010", mass="0.0001")

        sensors.append(ET.Element("framepos", objtype="site", objname=site_name, name=f"{self.name}_yoyo_pos"))
        sensors.append(ET.Element("framelinvel", objtype="site", objname=site_name, name=f"{self.name}_yoyo_vel"))
        sensors.append(ET.Element("framequat", objtype="site", objname=site_name, name=f"{self.name}_yoyo_quat"))
        sensors.append(ET.Element("frameangvel", objtype="site", objname=site_name, name=f"{self.name}_yoyo_ang_vel"))
        return bumblebee_xml

    def bind_to_data(self, data: mujoco.MjData) -> None:
        super().bind_to_data(data)
        self.sensors["yoyo_pos"] = self.data.sensor(f"{self.name}_yoyo_pos")
        self.sensors["yoyo_vel"] = self.data.sensor(f"{self.name}_yoyo_vel")
        self.sensors["yoyo_quat"] = self.data.sensor(f"{self.name}_yoyo_quat")
        self.sensors["yoyo_ang_vel"] = self.data.sensor(f"{self.name}_yoyo_ang_vel")

    def update(self) -> None:
        
        if (self.switch==1 or len(self.h_opt_interp)==0):
            self.update_control()
       

        if self.time < self.max_computing_time and (len(self.h_opt_interp_before)!=0):
            self.h = self.h_opt_interp_before[self.index_control]
            self.h_dot = self.h_dot_opt_interp_before[self.index_control]
            self.h_ddot = self.h_ddot_opt_interp_before[self.index_control]
        else:
            self.h = self.h_opt_interp[self.index_control]
            self.h_dot = self.h_dot_opt_interp[self.index_control]
            self.h_ddot = self.h_ddot_opt_interp[self.index_control]


        self.index_control=self.index_control+1
        self.time=self.time+self.dt
       
        if (self.theta<0 and self.switch==0):
            self.switch=1
            self.index_control=0
            self.time=0
        else:
            self.switch=0
       
        self.update_yoyo_state()       
        
        setpoint = {"load_mass": 0.0, # [m] - assume no load mass (rope tension will account for yoyo)
                  "target_pos": np.array([0, 0, self.h]), # [m] - desired hand position
                  "target_vel": np.array([0, 0, self.h_dot]), # [m/s] - desired hand velocity
                  "target_acc": np.array([0, 0, self.h_ddot]), # [m/s^2] - desired hand acceleration
                  "target_rpy": np.zeros(3), # [rad] - (roll, pitch, yaw) not needed
                  "target_ang_vel": np.zeros(3)} # [rad/s] - (roll, pitch, yaw) not needed

        self.data.qfrc_applied[2] = -self.T # [N] - update forces every cycle [12 dimensions]

        self.data.qpos[11] = - self.d

        self.ctrl_output = self.controller.compute_control(state=self.state, setpoint=setpoint)
        motor_thrusts = self.input_matrix @ self.ctrl_output
        thrustsum = 0
        for propeller, thrust in zip(self.propellers, motor_thrusts):
            propeller.ctrl[0] = thrust
            thrustsum += thrust
        self.save_drone_state_to_csv(thrustsum)
       
    def save_drone_state_to_csv(self,thrustsum):
        cur_pos = self.state['pos']
        cur_vel = self.state['vel']
        cur_acc = self.state['acc']
        head=["Theta", "Theta_dot", "Theta_ddot", "h", "h_dot", "h_ddot", "pos", "vel", "acc", "tension", "d", "f", "z", "z_dot", "z_ddot"]
        self.csvdata.append([self.theta, self.theta_dot, self.theta_ddot, self.h, self.h_dot, self.h_ddot, cur_pos[2], cur_vel[2], cur_acc[2]-9.81, self.T, self.d, thrustsum, self.z, self.z_dot, self.z_ddot])

        if self.index_write == 0:
            with open('drone_state.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(head)

        if self.index_write % 500 == 0 and self.index_write != 0:
            with open('drone_state.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.csvdata)
                self.csvdata=[]

        self.index_write+=1
    def update_yoyo_state(self):
         # Yoyo dynamics
        self.r_theta = YoyoBumblebee1DOF_MPC.r_0 + (self.theta * YoyoBumblebee1DOF_MPC.alpha)
        cur_pos = self.state['pos']
        cur_vel = self.state['vel']
        cur_acc = self.state['acc']
        if self.theta <= 0 and self.theta_dot < 0:
            self.T_imp = round((2 * 3.14)/(abs(self.theta_dot) * (1 + YoyoBumblebee1DOF_MPC.beta)) / self.dt,0)
            self.theta_dot = - YoyoBumblebee1DOF_MPC.beta * self.theta_dot
            self.yoyo_z = cur_pos[2] - 2.2 - YoyoBumblebee1DOF_MPC.L
            self.T = YoyoBumblebee1DOF_MPC.m_yoyo * ((((1 + YoyoBumblebee1DOF_MPC.beta) * abs(self.theta_dot) * YoyoBumblebee1DOF_MPC.r_0) / (self.dt * self.T_imp)) + cur_acc[2] - 9.81)
        else:
          
            self.T = YoyoBumblebee1DOF_MPC.m_yoyo * (cur_acc[2] + (YoyoBumblebee1DOF_MPC.alpha * self.theta_dot * self.theta_dot) * (self.r_theta * self.theta_ddot))
            self.theta_ddot = - (self.T * self.r_theta / YoyoBumblebee1DOF_MPC.J_yoyo)
            self.theta_dot += self.theta_ddot * self.dt
        


        # Update Yoyo state
        self.theta += self.theta_dot * self.dt

        self.z = cur_pos[2] - YoyoBumblebee1DOF_MPC.L + (YoyoBumblebee1DOF_MPC.r_0 * self.theta) + ((YoyoBumblebee1DOF_MPC.alpha / 2) * (self.theta * self.theta))
        self.z_dot = cur_vel[2] + (self.r_theta * self.theta_dot)
        self.z_ddot = (cur_acc[2] - 9.81) + (YoyoBumblebee1DOF_MPC.alpha * self.theta_dot * self.theta_dot) + (self.r_theta * self.theta_ddot)

        self.d = cur_pos[2] - self.z
    def update_control(self):
        x0_val=[self.theta,self.theta_dot,self.h,self.h_dot]
        self.h_opt_interp_before = list(self.h_opt_interp)
        self.h_dot_opt_interp_before = list(self.h_dot_opt_interp)
        self.h_ddot_opt_interp_before = list(self.h_ddot_opt_interp)
        w_opt=control.controller(x0_val,parameters.x0_init,parameters.theta_dot_ref,parameters.N,parameters.M)
        [_,_,self.h_opt,self.h_dot_opt,self.h_ddot_opt,self.dt_opt]=support.disassemble(w_opt)    
        parameters.x0_init=w_opt
        
        [self.h_opt_interp,self.h_dot_opt_interp,self.h_ddot_opt_interp]=support.interpolation(self.h_opt,self.h_dot_opt,self.h_ddot_opt,self.dt_opt,0.001,parameters.N,parameters.M)
       
    def _compute_in_subprocess(self, q: Queue):
        self.compute_control()

        q.put((
        self.h_opt_interp,
        self.h_dot_opt_interp,
        self.h_ddot_opt_interp
        ))