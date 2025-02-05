
#add relevant imports 
import jax
from jax import numpy as jp

# Weights for rewards 
WEIGHTS = {
    'velocity': 2,
    'base_height': 3,
    'control_cost': -3, # penalize control actions
    'rotation_cost': 0, # right now rotatation rewards aren't being given 
    'upright_reward': 2
}


# Dictionary with Sensor Data 
SENSOR_END_IDX = {
    'position' : 3,
    'gyro' : 6,
    'local_linvel' : 9,
    'accelerometer' : 12,
    'upvector': 15,
    'forwardvector' : 18,
    'left_foot_force' : 21,
    'right_foot_force' : 24
}

# List of Feet geoms 
LEG_GEOMS = ['left_foot', 'right_foot', 'left_shin', 'right_shin', 'left_thigh', 'right_thigh'] 

CONTROLLED_ACTUATORS = {
    "right_hip_x": 0,
    "right_hip_y": 1,
    "right_hip_z": 2,
    "right_knee": 3,
    "right_foot": 4,
    "left_hip_x": 5,
    "left_hip_y": 6,
    "left_hip_z": 7,
    "left_knee": 8,
    "left_foot": 9,
    # "right_shoulder1": 10,
    # "right_shoulder2": 11,
    # "right_elbow": 12,
    # "left_shoulder1": 13,
    # "left_shoulder2": 14,
    # "left_elbow": 15
}

NUM_CONTROL_JOINTS = 10 # number of controlled joints