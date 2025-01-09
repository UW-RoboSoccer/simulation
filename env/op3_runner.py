"""Environment for the running OP3 model."""

import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax import actuator
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

OP3_SCENE_PATH = Path('.') / "assets" / "op3" / "scene.xml"
HUMANOID_SCENE_PATH = Path('.') / "assets" / "humanoid" / "humanoid_pos.xml"


class OP3Runner(PipelineEnv):
    """Environment for the running OP3 model."""
    
    def __init__(
            self,
            **kwargs
    ):
        path = kwargs.pop('path', OP3_SCENE_PATH)
        mj_model = mujoco.MjModel.from_xml_path(str(path))

        self.sensor_data = {}
        self.mj_data = mujoco.MjData(mj_model)

        sys = mjcf.load_model(mj_model)
        self.sys = sys

        self.sensor_data = {'accel':[],
                            'gyro':[]}
        self.lfootid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'leftFoot')
        self.rfootid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'rightFoot')

        print(self.lfootidm, self.rfootid)

        print(len(self.sys.init_q))
        print(self.sys.init_q)
   
        
        super().__init__(sys, backend='mjx', **kwargs)

    def reset(self, rng: jp.ndarray) -> base.State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2 = jax.random.split(rng, 3)

        qpos = self.sys.init_q + jax.random.uniform(rng1, self.sys.q_size(), minval=-1, maxval=1, dtype=jp.float32)
        qvel = jax.random.uniform(rng2, self.sys.qd_size(), minval=-1, maxval=1, dtype=jp.float32)

        # Reset actuator activation states
        act = jp.zeros(self.sys.act_size())

        pipeline_state = self.pipeline_init(qpos, qvel) # takes in q, qd, act, ctrl, returns base.State with args: q, qd, x, xd
        # print(len(pipeline_state.x.pos))
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size(), dtype=jp.float32))
        reward, done, zero = jp.zeros(3, dtype=jp.float32)
        metrics = {
            'reward_linup': zero,
            'reward_linvel': zero,
        }

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""

        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min    # rescale action to be within ctrl_range

        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action) # returns base.State with args: q, qd, x, xd

        return
    

    def _get_obs(self, pipeline_state: base.State, action: jp.ndarray) -> jp.ndarray:
        """Observes humanoid positions, velocities, and angles."""
        position = pipeline_state.q
        velocity = pipeline_state.qd
        time_step = self.dt

        self.sensor_data['gyro']= self.mj_data.sensor('gyro').data.copy()
        self.sensor_data['accel'] = self.mj_data.sensor('accel').data.copy()

        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jp.hstack(
            [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
        )

        xd_i = (
            base.Transform.create(pos=x_i.pos - pipeline_state.x.pos)
            .vmap()
            .do(pipeline_state.xd)
        )

        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_ang = xd_i.ang
        com_velocity = jp.hstack([com_vel, com_ang])

        qfrc_actuator = actuator.to_tau(
            self.sys, action, pipeline_state.q, pipeline_state.qd
        )

        return jp.concatenate([
            position,
            velocity,
            com_inertia.ravel(),
            com_velocity.ravel(),
            qfrc_actuator,
        ])


    def _com(self, pipeline_state: base.State) -> jp.ndarray:
        """Returns the center of mass position."""
        inertia = self.sys.link.inertia         # Link(Base): rigid segment of an articulated body, link.inertia: Inertia, Inertia(Base): angular inertia, mass and com location
        mass_sum = jp.sum(inertia.mass)

        x_i = pipeline_state.x.vmap().do(inertia.transform) # pipeline_state.x(Transform) : link position in world frame, inertia.transform: com position and orientation
                                                            # .do() : applies the position and quaternion calculations to the links, based on link position in world frame
                                                            # and 
        com = (
            jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i
    

    def _get_contact_forces(self, piipline_state: base.State) -> jp.ndarray:
        """Returns the contact forces."""

    
    # def height_reward(self, )


if __name__ == '__main__':
    rng = jax.random.PRNGKey(0)

    env = OP3Runner()
    state = env.reset(rng)
    print("Initial state:")
    print("Reward: ", state.reward)
    print("Observation: ", state.obs)
    print("Done: ", state.done)
    print("Metrics: ", state.metrics)


