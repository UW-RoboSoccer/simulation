
import brax
from brax.envs.base import PipelineEnv, State
from brax import envs
from brax.io import mjcf
from brax import base
from brax.base import Transform
from brax import math
import time 

import brax.math
import jax
from jax import numpy as jp

import mujoco
from mujoco import mjx
from mujoco.mjx._src import support

import os

from envs.walking_rewards import _base_height_reward, _velocity_reward, _rotating_reward, _upright_reward, _control_actions_reward

WEIGHTS = {
    'velocity': 2,
    'base_height': 5,
    'control_cost': -3, # penalize control actions
    'rotation_cost': 0, # right now rotatation rewards aren't being given 
    'upright_reward': 5
}


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



class HumanoidWalker(PipelineEnv):
    def __init__(self, **kwargs):

        path = os.path.join(os.path.dirname(__file__), "..", "assets", "humanoid", "ernest_humanoid_mod.xml")
        mj_model = mujoco.MjModel.from_xml_path(str(path))

        self.mj_data = mujoco.MjData(mj_model)
        self.mjx_model = mjx.put_model(mj_model)

        self.sys = mjcf.load_model(mj_model)

        self.reset_noise_scale = 1e-2

        self.seed = jax.random.PRNGKey(0)

        super().__init__(self.sys, backend='mjx', **kwargs)

        # indeces of data
        self.accel_end_idx = 3
        self.gyro_end_idx = 6
        self.lin_accel_end_idx = 9


        self.target_height = 0.45
        self.roll_weight = 1.0
        self.pitch_weight = 0.8
        self.yaw_weight = 0.6

        #IMU Sensor and Contact Force
        self.num_sensors = sensor_end_idx['right_foot_force'] - 1
        self.sensor_data = jp.zeros(self.num_sensors)


        #Define size of arrays returned with sensor data
        self.sensor_sizes = {
            'position' : 3,
            'gyro' : 3,
            'local_linvel' : 3,
            'accelerometer' : 3,
            'upvector': 3,
            'forwardvector' : 3,
            'left_foot_force' : 3,
            'right_foot_force' : 3
        }



    def reset(self, rng: jp.ndarray) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self.reset_noise_scale, self.reset_noise_scale

        humanoid_qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), jp.float32, minval=low, maxval=hi
        )

        humanoid_qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )

        act = jp.zeros(self.sys.act_size(), jp.float32)

        #concatenate ball and goal positions and velocites into this below
        qpos = jp.concatenate([humanoid_qpos])
        qvel = jp.concatenate([humanoid_qvel])

        pipeline_state = self.pipeline_init(qpos, qvel, act)

        obs = self._get_obs(pipeline_state, act)



        reward, done, zero = jp.zeros(3)

        metrics = {
            'total_reward': zero,
            'velocity_reward': zero,
            'base_height_reward': zero,
            'control_cost': zero,
            'rotation_cost': zero,
            'upright_reward': zero,
        }

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:

        #normalize actions
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min



        prev_q = state.pipeline_state.q
        #get pipeline state
        pipeline_state = self.pipeline_step(state.pipeline_state, action)


        #Get up to date observations
        obs = self._get_obs(pipeline_state, action)

        weighted_rewards, reward, done = self._calculate_reward(pipeline_state=pipeline_state, sensor_data=obs,target_height=self.target_height, 
                                                           prev_q=prev_q)

        metrics = {
            'total_reward': reward,
            'velocity_reward': weighted_rewards[0],
            'base_height_reward': weighted_rewards[1],
            'control_cost':  weighted_rewards[2],
            'rotation_cost': weighted_rewards[3],
            'upright_reward': weighted_rewards[4],
        }


        #Replace data in state with newly acquired data
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, metrics=metrics)


    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:

        #get sensor data
        sensor_data = pipeline_state.sensordata
        return sensor_data


    def _calculate_reward(self, pipeline_state, sensor_data, target_height, prev_q) -> float:
        "Calculate Rewards"

        all_rewards = {
            'velocity' : _velocity_reward(pipeline_state),
            'base_height' : _base_height_reward(sensor_data, target_height),
            'control_cost': _control_actions_reward(pipeline_state, prev_q),
            'rotation_cost': _rotating_reward(sensor_data, self.roll_weight, self.pitch_weight, self.yaw_weight),
            'upright_reward': _upright_reward(sensor_data)
        }
            # Calculate weighted sum of rewards

        REWARD_WEIGHT = jp.array([WEIGHTS[key] for key in all_rewards.keys()])
        reward_values = jp.array([all_rewards[key] for key in all_rewards.keys()])


        weighted_rewards = REWARD_WEIGHT * reward_values
        reward = jp.sum(weighted_rewards)
        done = 1 - weighted_rewards[1] # if the robot is 'not standing' we want to terminate the ep

        return weighted_rewards, reward, done
    
# Code to test compilation of the environment 

envs.register_environment('ernest_walker', HumanoidWalker)

env = HumanoidWalker()
rng = jax.random.PRNGKey(0)

state = env.reset(rng)
obs = env._get_obs(state.pipeline_state, jp.zeros(env.sys.act_size()))

print("Initial State:")
