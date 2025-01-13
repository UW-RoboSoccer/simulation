import brax
from brax.envs.base import PipelineEnv, State
from brax import envs
from brax.io import mjcf
from brax import base
from brax.base import Transform

import brax.math
import jax
from jax import numpy as jp

import mujoco
from mujoco import mjx
from mujoco.mjx._src import support

import os

class HumanoidKick(PipelineEnv):
    def __init__(self, **kwargs):
        
        path = os.path.join(os.path.dirname(__file__), "..", "assets", "humanoid", "humanoid_pos.xml")
        mj_model = mujoco.MjModel.from_xml_path(str(path))
        self.mj_data = mujoco.MjData(mj_model)
        self.mjx_model = mjx.put_model(mj_model)

        sys = mjcf.load_model(mj_model)
        self.sys = sys

        super().__init__(self.sys, backend='mjx', **kwargs)

        self.sensorData = { 'accel': [],
                            'gyro' : []}

        self.links = self.sys.link_names
        print(self.links)

        self.sensorData['accel'].append(self.mj_data.sensor('accel').data.copy())
        self.sensorData['gyro'].append(self.mj_data.sensor('gyro').data.copy())
        print(self.sensorData)

        
    #in another file, define helper functions like get_link_index and import
    # def _get_link_index(self, name):
    #     return self.sys.find_name(name)

    def reset(self, rng: jp.ndarray) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        humanoid_qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), jp.float32, minval=-1, maxval=1
        )

        humanoid_qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=-1, maxval=1
        )

        # reset ball and target when implemented later

        qpos = jp.concatenate([humanoid_qpos])
        qvel = jp.concatenate([humanoid_qvel])

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)

        metrics = {
            'stabilizeReward': 0,
            'kickReward': 0,
        }
        print('end of reset function')
        done = True
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        obs = self._get_obs(pipeline_state, action)

        # kickReward, doneKick = calculate_kick_reward(pipeline_state, action, obs)
        # standReward, doneStand = stabilization_reward(pipeline_state, action, obs)

        # done = jp.asarray(doneStand, dtype=jp.float32)
        done = False #for now
        reward = 10

        metrics = {
            'stabilizeReward': 0,
            'kickReward': 0,
        }

        state.metrics.update(metrics)

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        
        position = pipeline_state.q
        velocity = pipeline_state.qd

        #get sensor data
        self.sensorData['accel'].append(self.mj_data.sensor('accel').data.copy())
        self.sensorData['gyro'].append(self.mj_data.sensor('gyro').data.copy())

        #add actuator values to observation space

        # add target and ball positions

        lin_accel = self.sensorData['accel'][0]  
        ang_accel = self.sensorData['gyro'][0]

        # print('Linear Acceleration', lin_accel)
        # print('Angular Acceleration', ang_accel)


        # x = jp.concatenate([position, velocity, lin_accel, ang_accel])
        # print('observation space shape', x.size)

        # Combine arrays for output
        return jp.concatenate([
            position, 
            velocity,
            lin_accel,
            ang_accel,
            # Add actuators, ball, and target
        ])


envs.register_environment('kicker', HumanoidKick)

env = HumanoidKick()
rng = jax.random.PRNGKey(0)

state = env.reset(rng)
obs = env._get_obs(state.pipeline_state, jp.zeros(env.sys.act_size()))

print("Initial State:")
