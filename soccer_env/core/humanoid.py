from brax import envs
from brax.io import mjcf
from brax.envs.base import State, PipelineEnv
from jax import numpy as jp
from brax import base

import mujoco
import pathlib
import jax

class Humanoid(PipelineEnv):
    def __init__(self, foward_reward_w, alive_reward, ctrl_cost_w, terminate_when_dead, alive_z_range, **kwargs):
        path = pathlib.Path(__file__).parent / 'assets' / 'humanoid'
        mj_model = mujoco.MjModel.from_xml_path(path / 'humanoid.xml')

        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys=sys, **kwargs)

        self._foward_reward_w = foward_reward_w
        self._alive_reward = alive_reward
        self._ctrl_cost_w = ctrl_cost_w
        self._terminate_when_dead = terminate_when_dead
        self._alive_z_range = alive_z_range

    def step(self, state: State, action: jp.ndarray) -> State:
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = action_min + (action_max - action_min) * (action + 1.0) / 2.0 # Normalize action to [-1, 1]

        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        center_of_mass0, *_ = self._com(pipeline_state0)
        center_of_mass, *_ = self._com(pipeline_state)

        min_z, max_z = self._alive_z_range
        alive = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        alive = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, alive)

        ctrl_cost = self._ctrl_cost_w * jp.sum(jp.square(action))

    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = (
            jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i
