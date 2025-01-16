import jax
import jax.numpy as jp
from brax import math
from brax.base import State
from typing import Dict, Tuple

WEIGHTS = {
    'velocity': 0.30,
    'feet_contact': 0.20,
    'base_height': 0.20,
    'base_acceleration': 0.15,
    'action_difference': 0.10,
}

##Modify to only use: (done)velocity, (done)base acceleration, (done)feet contact, (done)base height, and action difference

@jax.jit
def _velocity_reward(sensor_data, target_vel: jp.ndarray, is_standing: bool) -> jp.ndarray:
    """Compute velocity reward based on x, y velocity difference from target."""    
    
    #ideal velocity is 0
    # jax.debug.print("gyro data in reward: {}", sensor_data['gyro'])
    vel_diff = sensor_data['gyro']
    vel_diff_mag = jp.sqrt(jp.sum(jp.square(vel_diff)))
    # jax.debug.print("Vel diff: {}", vel_diff)

    return jp.where(
        is_standing,
        jp.exp(-5 * vel_diff_mag),
        jp.exp(-5 * jp.square(vel_diff_mag))
    )
    
@jax.jit
def _base_height_reward(obs: jp.ndarray, target_height: float = 1.0) -> jp.ndarray:
    """Computer base height reward"""
    current_height = obs[2]
    return jp.exp(-20 * jp.abs(current_height - target_height))

@jax.jit
def _base_acceleration_reward(sensor_data):
    """Compute base acceleration reward"""
    accel_data = sensor_data['accel']
    return jp.exp(-0.01 * jp.sum(jp.abs(accel_data)))

@jax.jit
def _feet_contact_reward(sensor_data):
    """Compute feet contact reward"""

    has_contact = jp.logical_or(
        jp.any(sensor_data['left_contact'] > 0),
        jp.any(sensor_data['right_contact'] > 0)
    )

    return jp.where(
        has_contact,
        1.0,
        0.0
    )

@jax.jit
def _action_difference_reward(action: jp.ndarray, prev_action: jp.ndarray, ctrl_range: jp.ndarray) -> jp.ndarray:
    """Compute action difference reward"""
    action_min = ctrl_range[:, 0]
    action_max = ctrl_range[:, 1]

    action = (action + 1) * (action_max - action_min) * 0.5 + action_min
    prev_action = (prev_action + 1) * (action_max - action_min) * 0.5 + action_min

    action_diff = jp.sum(jp.abs(action - prev_action))

    return jp.exp(-0.02 * action_diff)

# @jax.jit
def _calculate_reward( pipeline_state: State, sensor_data, is_standing, target_height) -> float:
    """Calculate reward"""

    all_rewards = {
        'velocity' : _velocity_reward(sensor_data, target_height, is_standing),
        'base_height' : _base_height_reward(pipeline_state.q, target_height),
        'base_acceleration' : _base_acceleration_reward(sensor_data),
        'feet_contact' : _feet_contact_reward(sensor_data),
        'action_difference' : _action_difference_reward(jp.zeros(8), jp.zeros(8), jp.zeros((8, 2))) ##CHANGE THIS
    }

    # Calculate weighted sum of rewards
    reward = sum([WEIGHTS[key] * all_rewards[key] for key in all_rewards.keys()])

    return reward, all_rewards




