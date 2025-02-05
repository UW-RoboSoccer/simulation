
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

import envs.walking_consts as consts 
import envs.walking_rewards as rewards


class HumanoidWalker(PipelineEnv):
    def __init__(self, **kwargs):

        path = os.path.join(os.path.dirname(__file__), "..", "assets", "humanoid", "ernest_humanoid.xml")
        mj_model = mujoco.MjModel.from_xml_path(str(path))

        self.mj_data = mujoco.MjData(mj_model)
        self.mjx_model = mjx.put_model(mj_model)

        self.sys = mjcf.load_model(mj_model)

        self.reset_noise_scale = 1e-2

        self._controlled_joints = consts.NUM_CONTROL_JOINTS

        jax.debug.print("qspace {}", self.sys.init_q)

        self.seed = jax.random.PRNGKey(0)

        super().__init__(self.sys, backend='mjx', **kwargs)

        self.target_height = 0.45
        self.roll_weight = 1.0
        self.pitch_weight = 0.8
        self.yaw_weight = 0.6

        #IMU Sensor and Contact Force
        self.num_sensors = consts.SENSOR_END_IDX['right_foot_force'] - 1
        self.sensor_data = jp.zeros(self.num_sensors)
        


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

        full_action = jp.zeros(self.sys.act_size())

        for i, (_, actuator_idx) in enumerate(consts.CONTROLLED_ACTUATORS.items()):
            full_action = full_action.at[actuator_idx].set(action[i])

        full_action = (full_action + 1) * (action_max - action_min) * 0.5 + action_min

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
            'velocity' : rewards._velocity_reward(pipeline_state),
            'base_height' : rewards._base_height_reward(sensor_data, target_height),
            'control_cost': rewards._control_actions_reward(pipeline_state, prev_q),
            'rotation_cost': rewards._rotating_reward(sensor_data, self.roll_weight, self.pitch_weight, self.yaw_weight),
            'upright_reward': rewards._upright_reward(sensor_data)
        }
            # Calculate weighted sum of rewards

        REWARD_WEIGHT = jp.array([consts.WEIGHTS[key] for key in all_rewards.keys()])
        reward_values = jp.array([all_rewards[key] for key in all_rewards.keys()])


        weighted_rewards = REWARD_WEIGHT * reward_values
        reward = jp.sum(weighted_rewards)
        done = 1 - weighted_rewards[1] # if the robot is 'not standing' we want to terminate the ep

        return weighted_rewards, reward, done
    
    #  # Change the action size to be only the number of controlled joints for the RL policy
    @property
    def action_size(self) -> int:
        return self._controlled_joints
    
# Code to test compilation of the environment 

envs.register_environment('ernest_walker', HumanoidWalker)

env = HumanoidWalker()
rng = jax.random.PRNGKey(0)

state = env.reset(rng)
obs = env._get_obs(state.pipeline_state, jp.zeros(env.sys.act_size()))

print("Initial State:")
