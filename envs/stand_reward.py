import jax
import jax.numpy as jp
from brax import math
from brax.base import State
from typing import Dict, Tuple

WEIGHTS = {
    'velocity': 0.285,
    'feet_contact': 0.20,
    'base_height': 0.275,
    'base_acceleration': 0.10,
    'action_difference': 0.09,
    'upright' : 0.01
}

##Modify to only use: (done)velocity, (done)base acceleration, (done)feet contact, (done)base height, and action difference

@jax.jit
def _velocity_reward(sensor_data, target_vel: jp.ndarray, is_standing: bool) -> jp.ndarray:
    """Penalize downwards velocity"""

    downward_velocity = jp.array(sensor_data['linear_velocity'][2])

    #ideal velocity is 0
    # jax.debug.print("gyro data in reward: {}", sensor_data['gyro'])
    # vel_diff = sensor_data['gyro']
    # velocity_direction = jp.sign(downward_velocity)

    vel_diff_mag = jp.sqrt(jp.sum(jp.square(downward_velocity)))
    # jax.debug.print("Vel diff: {}", vel_diff)

    return jp.exp(-5 * (vel_diff_mag + 1))
    
@jax.jit
def _base_height_reward(obs: jp.ndarray, target_height: float = 1.0) -> jp.ndarray:
    """Computer base height reward"""
    
    threshold = 0.2
    current_height = obs[2]

    x = jp.abs(target_height - current_height)

    return jp.where( x < threshold,
        0, #-2.5 * (x - threshold) * (x + threshold),
        -20 * (x - threshold) * (x + threshold)
    )
    
@jax.jit
def _base_acceleration_reward(sensor_data):
    """Compute base acceleration reward"""
    accel_data = sensor_data['linear_acceleration']
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

@jax.jit
def _upright(pipeline_state: State) -> jp.ndarray:
    """Compute uprightness reward"""
    q = pipeline_state.q[3:7]
    k = jp.array([0.0, 0.0, 1.0])
    up = math.rotate(k, q)
    projection = jp.dot(up, k)
    return jp.tan(projection * (4.65 / jp.pi))

def _calculate_reward( pipeline_state: State, sensor_data, is_standing, target_height,
                      action: jp.ndarray, prev_action: jp.ndarray, ctrl_range : jp.ndarray) -> float:
    """Calculate reward"""

    all_rewards = {
        'velocity' : _velocity_reward(sensor_data, target_height, is_standing),
        'base_height' : _base_height_reward(pipeline_state.q, target_height),
        'base_acceleration' : _base_acceleration_reward(sensor_data),
        'feet_contact' : _feet_contact_reward(sensor_data),
        'action_difference' : _action_difference_reward(action, prev_action, ctrl_range),
        'upright' : _upright(pipeline_state)
    }

    # Calculate weighted sum of rewards
    REWARD_WEIGHT = jp.array([WEIGHTS[key] for key in all_rewards.keys()])
    reward_values = jp.array([all_rewards[key] for key in all_rewards.keys()])

    reward = jp.sum(REWARD_WEIGHT * reward_values)


    return reward, all_rewards


##Negatively reward base height
##Negatively reward downwards velocity
