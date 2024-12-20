import os

from brax.io import mjcf
from brax import base
from brax import actuator
from brax.envs.base import PipelineEnv, State
from brax import math 

import mujoco
from mujoco import mjx

import jax
from jax import numpy as jp

# Constant Definitions
POSITION_QUATERNION_SIZE = 7 # = xyz (3) + quaternion (4)
VELOCITY_SIZE = 6 # = linear (3) + angular velocity (3)


class Environment(PipelineEnv):
    def __init__(self, **kwargs):
        #Importing the model
        path = os.path.join(os.path.dirname(__file__), "..", "assets", "kicking", "op3_simplified.xml" )
        mj_model = mujoco.MjModel.from_xml_path(str(path))
        self.mj_data = mujoco.MjData(mj_model)
        sys = mjcf.load_model(mj_model)
        self.sys = sys
        self.sensorData = {'accel' : [],
                            'gyro' : []}

        self.lFootid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'leftFoot')
        self.rFootid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'rightFoot')
        self.links = self.sys.link_names

        #define class members for indexing different entities in the simulation

        super().__init__(sys, backend='mjx', **kwargs)


    def reset(self, rng: jp.ndarray) -> State:

        return State(pipeline_state, obs, reward, done, metrics)


    def step(self, state: State, action: jp.ndarray) -> State:
        #Place Holders
        pipeline_state = state.pipeline_state
        obs = None
        reward = None
        done = None
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(
        self, pipeline_state: base.State, action: jax.Array
    ) -> jax.Array:
        """Observes humanoid body position, velocities, and angles."""
        """Observe contact forces, sensor data, and properties of other bodies"""

        return jp.concatenate([
            position,
            velocity,
            self.sensorData,
            # qfrc_actuator,
            goal_pos,
            ball_pos,
            ball_vel,
        ])

    def _get_contact_forces(self, pipeline_state: base.State) -> jax.Array:
        #Fill this out
        return None




