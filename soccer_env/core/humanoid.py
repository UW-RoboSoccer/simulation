import mujoco
import pathlib
import jax
import os

from brax import envs
from brax.io import mjcf
from brax.envs.base import State, PipelineEnv
from jax import numpy as jp
from brax import base
from brax import actuator 


class Humanoid(PipelineEnv):
    def __init__(self, forward_reward_w, alive_reward, ctrl_cost_w, terminate_when_dead, alive_z_range, _exclude_current_positions_from_observation=True,  reset_noise_scale=1e-2, **kwargs):
        path = pathlib.Path(__file__).parent.parent / 'assets' / 'op3'
        print(path)
        mj_model = mujoco.MjModel.from_xml_path(os.path.join(path, 'op3.xml'))
            # path / 'op3.xml')

        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys=sys, **kwargs)

        self._forward_reward_w = forward_reward_w
        self._alive_reward = alive_reward
        self._ctrl_cost_w = ctrl_cost_w
        self._terminate_when_dead = terminate_when_dead
        self._reset_noise_scale = reset_noise_scale
        self._alive_z_range = alive_z_range
        self._exclude_current_positions_from_observation = _exclude_current_positions_from_observation

    def reset(self, rng: jp.ndarray) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
    
        qpos = self.sys.init_q + jax.random.uniform(rng1,  (self.sys.q_size(),), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=low, maxval=hi)

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        metrics = {
            'forward_reward': zero,
            # 'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = action_min + (action_max - action_min) * (action + 1.0) / 2.0 # Normalize action to [-1, 1]

        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        center_of_mass0, *_ = self._com(pipeline_state0)
        center_of_mass, *_ = self._com(pipeline_state)
        velocity = (center_of_mass.pos - center_of_mass0.pos) / self.dt

        min_z, max_z = self._alive_z_range
        alive = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        alive = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, alive)

        if self._terminate_when_dead: 
            alive_reward = self._alive_reward
        else: 
            alive_reward = self._alive_reward * alive 

        forward_reward = self._foward_reward_w * velocity[0]
        ctrl_cost = self._ctrl_cost_w * jp.sum(jp.square(action))

        obs = self._get_obs(pipeline_state, action)

        reward = forward_reward + alive_reward - ctrl_cost
        done = 1.0 - alive if self._terminate_when_dead else 0.0
        
        state.metrics.update(
            forward_reward=forward_reward,
            # reward_linvel=forward_reward, # wut is htis for 
            reward_quadctrl=-ctrl_cost, 
            reward_alive=alive,
            x_position=center_of_mass[0],
            y_position=center_of_mass[1],
            distance_from_origin=jp.linalg.norm(center_of_mass),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
        )   

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(
      self, pipeline_state: base.State, action: jax.Array
    ) -> jax.Array:
        position = pipeline_state.q 
        velocity = pipeline_state.qd

        if self._exclude_current_positions_from_observation: # prevents dependence on curr abs pos
            position = position[2:]

        center_of_mass, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - center_of_mass).vmap().do(inertia) # compute rotational inertia about center of mass
        center_of_mass_inertia = jp.hstack([cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]])

        xd_i = ( 
            base.Transform.create(pos=x_i.pos - pipeline_state.x.pos)
            .vmap()
            .do(pipeline_state.xd)
        )

        center_of_mass_speed = inertia.mass[:, None] * xd_i.vel / mass_sum
        center_of_mass_angle = xd_i.ang 
        center_of_mass_velocity = jp.hstack([center_of_mass_speed, center_of_mass_angle])

        qfrc_actuator = actuator.to_tau(
            self.sys, action, pipeline_state.q, pipeline_state.qd
        )

        return jp.concatenate([ 
            position, 
            velocity, 
            center_of_mass_inertia.ravel(), 
            center_of_mass_velocity.ravel(),

        ])


    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = (
            jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i
