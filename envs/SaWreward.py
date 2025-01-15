import jax
import jax.numpy as jp
from brax import math
from brax.base import State
from typing import Dict, Tuple

WEIGHTS = {
    'velocity': 0.35,
    'feet_contact': 0.25,
    'base_height': 0.15,
    'base_acceleration': 0.15,
    'action_difference': 0.10,
}

##Modify to only use: (done)velocity, (done)base acceleration, (done)feet contact, (done)base height, and action difference

@jax.jit
def velocity_reward(pipeline_state: State, target_vel: jp.ndarray, is_standing: bool) -> jp.ndarray:
    """Compute velocity reward based on x, y velocity difference from target.
    Args:
        qd: Current velocity [2,] (x,y)
        target_vel: Target velocity [2,]
        is_standing: Whether the agent is standing
    """    
    vel_diff = pipeline_state.qd[:2] - target_vel
    return jp.where(
        is_standing,
        jp.exp(-5 * vel_diff),
        jp.exp(-5 * jp.square(vel_diff))
    )
    
@jax.jit
def base_height_reward(obs: jp.ndarray, target_height: float = 1.0) -> jp.ndarray:
    """Computer base height reward"""
    current_height = obs[2]
    return jp.exp(-20 * jp.abs(current_height - target_height))

@jax.jit
def base_acceleration_reward(sensorData):
    """Compute base acceleration reward"""
    accel_data = sensorData['accel']
    return jp.exp(-0.01 * jp.sum(jp.abs(accel_data)))

@jax.jit
def feet_contact_reward(sensorData):
    """Compute feet contact reward"""
    reward = 0
    if sensorData['left_contact'] or sensorData['right_contact']:
        reward += 1
    return reward

@jax.jit
def action_difference_reward(action: jp.ndarray, prev_action: jp.ndarray, ctrl_range: jp.ndarray) -> jp.ndarray:
    """Compute action difference reward"""
    action_min = ctrl_range[:, 0]
    action_max = ctrl_range[:, 1]

    action = (action + 1) * (action_max - action_min) * 0.5 + action_min
    prev_action = (prev_action + 1) * (action_max - action_min) * 0.5 + action_min

    action_diff = jp.sum(jp.abs(action - prev_action))

    return jp.exp(-0.02 * action_diff)

@jax.jit
def calculate_reward( pipeline_state: State, sensorData, is_standing, target_height) -> float:
    """Calculate reward"""

    rewards = {
        'velocity' : velocity_reward(pipeline_state, 1, is_standing),
        'base_height' : base_height_reward(pipeline_state.q, target_height),
        'base_acceleration' : base_acceleration_reward(sensorData),
        'feet_contact' : feet_contact_reward(sensorData),
        'action_difference' : action_difference_reward(jp.zeros(8), jp.zeros(8), jp.zeros((8, 2))) ##CHANGE THIS

    }

    # Calculate weighted sum of rewards
    reward = sum([WEIGHTS[key] * rewards[key] for key in rewards.keys()])

    return reward




