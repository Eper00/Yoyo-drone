"""
This script shows how dynamic objects work.
"""

import os
import sys
import pathlib
import numpy as np

# make sure imports work by adding the necessary folders to the path:
project_root = pathlib.Path(__file__).parents[0]
sys.path.append(project_root.resolve().as_posix()) 

while "aiml_virtual" not in [f.name for f in  project_root.iterdir()]:
    project_root = project_root.parents[0]
    sys.path.append(project_root.resolve().as_posix())

import aiml_virtual
xml_directory = aiml_virtual.xml_directory
from aiml_virtual import scene, simulator
from aiml_virtual.simulated_object.dynamic_object.controlled_object.drone import yoyo_bumblebee_PD

if __name__ == "__main__":
    
    scn = scene.Scene(os.path.join(xml_directory, "empty_checkerboard.xml"), save_filename=f"result.xml")

    bby = yoyo_bumblebee_PD.YoyoBumblebee1DOF_PD()
    scn.add_object(bby, "0 0 1", "1 0 0 0", "0.5 0.5 0.5 1")
    
    sim = simulator.Simulator(scn)
  
    with sim.launch(fps=100,with_display=False):
        while sim.data.time<100:
            sim.tick()  