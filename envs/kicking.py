import brax
from brax.envs.base import PipelineEnv, State
from brax import envs
from brax.io import mjcf
from brax import base
from brax.base import Transform

from brax import math
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

        self.reset_noise_scale = 1e-2

        super().__init__(self.sys, backend='mjx', **kwargs)

        self.sensorData = { 'accel': [],
                            'gyro' : []}

        self.links = self.sys.link_names

        self.sensorData['accel'].append(self.mj_data.sensor('accel').data.copy())
        self.sensorData['gyro'].append(self.mj_data.sensor('gyro').data.copy())

        #contact forces
        self.floor_geom_id = support.name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.left_foot_geom_id = support.name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'left_foot')
        self.right_foot_geom_id = support.name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'right_foot')
        
    def reset(self, rng: jp.ndarray) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self.reset_noise_scale, self.reset_noise_scale

        humanoid_qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), jp.float32, minval=low, maxval=hi
        )

        humanoid_qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )

        # reset ball and target when implemented later

        #concatenate ball and target positions and velocites into this below
        qpos = jp.concatenate([humanoid_qpos])
        qvel = jp.concatenate([humanoid_qvel])

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)

        metrics = {
            'stabilizeReward': 0,
            'kickReward': 0,
            'accelerometer': jp.zeros(self.sensorData['accel'][0].shape),
            'gyroscope': jp.zeros(self.sensorData['gyro'][0].shape),
            'contactForces': jp.zeros(len(self.get_gait_contact(pipeline_state))),
        }
        # print('end of reset function')
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
            'accelerometer': self.sensorData['accel'][0],
            'gyroscope': self.sensorData['gyro'][0],
            'contactForces': self.get_gait_contact(pipeline_state),
        }


        state.metrics.update(metrics)
        print(state.metrics)

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        
        position = pipeline_state.q
        velocity = pipeline_state.qd

        #get sensor data
        self.sensorData['accel'].append(self.mj_data.sensor('accel').data.copy())
        self.sensorData['gyro'].append(self.mj_data.sensor('gyro').data.copy())

        #add actuator values to observation space

        contact_forces = self.get_gait_contact(pipeline_state)

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
    
    def get_gait_contact(self, pipeline_state: base.State):
        forces = jp.array([support.contact_force(self.mjx_model, self.mj_data, i) for i in range(self.mj_data.ncon)])

        right_contacts = pipeline_state.contact.geom == jp.array([self.floor_geom_id, self.right_foot_geom_id])
        left_contacts = pipeline_state.contact.geom == jp.array([self.floor_geom_id, self.left_foot_geom_id])

        right_contact_mask = jp.sum(right_contacts, axis=1) == 2
        left_contact_mask = jp.sum(left_contacts, axis=1) == 2

        total_right_forces = jp.sum(forces * right_contact_mask[:, None], axis=0)
        total_left_forces = jp.sum(forces * left_contact_mask[:, None], axis=0)

        return math.normalize(total_right_forces[:3])[1], math.normalize(total_left_forces[:3])[1]


envs.register_environment('kicker', HumanoidKick)

env = HumanoidKick()
rng = jax.random.PRNGKey(0)

state = env.reset(rng)
obs = env._get_obs(state.pipeline_state, jp.zeros(env.sys.act_size()))

# print("Initial State:")
