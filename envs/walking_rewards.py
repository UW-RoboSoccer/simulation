import jax
from jax import numpy as jp


sensor_end_idx = {
    'position' : 3,
    'gyro' : 6,
    'local_linvel' : 9,
    'accelerometer' : 12,
    'upvector': 15,
    'forwardvector' : 18,
    'left_foot_force' : 21,
    'right_foot_force' : 24
}

# ===== REWARD FUNCTIONS =====
# reward for height of robot
@jax.jit
def _base_height_reward(sensor_data: jp.ndarray, min_height: float = 0.4, max_height: float = 0.6) -> float:
    """Compute base height reward"""

    position = sensor_data[ sensor_end_idx['position'] - 3 : sensor_end_idx['position'] ]


    current_height = position[2]

    in_range = jp.where(current_height > max_height, 0.0, 1)
    in_range = jp.where(current_height < min_height, 0.0, in_range)

    return in_range

# reward for forward velocity of a robot
@jax.jit
def _velocity_reward(pipeline_state) -> jp.ndarray:
    """Reward linear velocity"""

    # extract the xyz coordinates of the torso linear velocity (global frame)
    linear_velocity = pipeline_state.qvel[0:3]
    return linear_velocity[0]

@jax.jit

def _control_actions_reward(pipeline_state, prev_q): 
    return jp.sum(jp.square(pipeline_state.q - prev_q))


@jax.jit
def _rotating_reward(sensor_data, roll_weight=1.0, pitch_weight=0.8, yaw_weight=0.6) -> jp.ndarray:
    """Compute uprightness reward"""
    angular_velocity = jp.array(sensor_data[ sensor_end_idx['gyro'] - 3 : sensor_end_idx['gyro'] ])

    roll_penalty = roll_weight * jp.abs(angular_velocity[0])
    pitch_penalty = pitch_weight * jp.abs(angular_velocity[1])
    yaw_penalty = yaw_weight * jp.abs(angular_velocity[2])

    return roll_penalty + pitch_penalty + yaw_penalty - 1 # prevent rewards from being too negative (- x - = +)


@jax.jit
def _upright_reward(sensor_data) -> jp.ndarray:
    upvector = jp.array(sensor_data[
        sensor_end_idx['upvector'] - 3 : sensor_end_idx['upvector']
    ])

    ideal_up = jp.array([0.0, 0.0, 1.0])  # World z-axis
    up_alignment = jp.dot(upvector, ideal_up)  # Ranges from -1 (upside down) to 1 (upright)

    # Exponential scaling to make it more sensitive near upright
    up_reward = jp.exp(5 * (up_alignment - 1))  # Adjust the exponent factor for sensitivity

    return up_reward

    #  model_path = "/content/drive/MyDrive/RoboCup/ernest_humanoid_og.xml"
