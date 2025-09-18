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

class YoyoBumblebee1DOF_MPC(bumblebee.Bumblebee):
    """
    Class that extends the Bumblebee to have a 1-DOF yoyo on it.
    """
    L = 0.6
    m_yoyo = 0.045
    r_0 = 0.005
    J_yoyo = 0.00003
    alpha = 0.0000165
    beta = 0.85

    theta = 0
    theta_dot = 100
    theta_ddot = 0
    d = 0
    r = 0



    T = 0
    z = 0
    z_dot = 0
    z_ddot = 0
    T_imp = 0

  
    dt = 0.001
    h = 1
    h_dot = 0
    h_ddot = 0
    k=0
    index=0
    csvdata=[]
    with open("MPC_input.csv", "r") as f:
      lines = f.readlines()

    # fejléc átugrása
    data_line = lines[1].strip()
    # csak az oszlopot választod ki, majd numpy array és reshape
    inputs = np.array([float(x) for x in data_line.split(",")]).reshape(1, -1)
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
        """
        Overrides drone.update. Updates the position of the propellers to make it look like they are
        spinning, and runs the controller.
        """
        self.spin_propellers()
        cur_pos = self.state['pos']
        cur_vel = self.state['vel']
        cur_acc = self.state['acc']
        self.r_theta = YoyoBumblebee1DOF_MPC.r_0 + (self.theta * YoyoBumblebee1DOF_MPC.alpha)
        self.h_ddot=self.inputs[0,self.k]
        self.k=self.k+1
        self.h += self.h_dot * self.dt
        self.h_dot += self.h_ddot * self.dt
        
        # Yoyo dynamics
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
        
        
        head=["Theta", "Theta_dot", "Theta_ddot", "h", "h_dot", "h_ddot", "pos", "vel", "acc", "tension", "d", "f", "z", "z_dot", "z_ddot"]
        self.csvdata.append([self.theta, self.theta_dot, self.theta_ddot, self.h, self.h_dot, self.h_ddot, cur_pos[2], cur_vel[2], cur_acc[2]-9.81, self.T, self.d, thrustsum, self.z, self.z_dot, self.z_ddot])

        if self.index == 0:
            with open('data_mujoco_30.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(head)

        if self.index % 500 == 0 and self.index/500 <= 2.55*2 and self.index != 0:
            with open('data_mujoco_30.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.csvdata)
                self.csvdata=[]

        self.index+=1