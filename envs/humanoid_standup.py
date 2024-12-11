from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax import base
from brax import actuator

import jax
from jax import numpy as jp

import pathlib

class HumanoidStandup(PipelineEnv):
    def __init__(self, **kwargs):
        # path = pathlib.Path(__file__).parent / 'assets\humanoid\humanoid.xml'
        path = r'C:\Users\ethan\robocup\soccer_env\assets\humanoid\humanoid.xml'
        sys = mjcf.load(str(path))

        super().__init__(sys, backend='mjx', **kwargs)

    def reset(self, rng: jp.ndarray) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -0.01, 0.01
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        metrics = {
            'reward_linup': zero,
            'reward_quadctrl': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)
    
    def step(self, state: State, action: jp.ndarray) -> State:
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]

        action = (action + 1) * (action_max - action_min) * 0.5 + action_min # map action from [-1, 1] to [action_min, action_max]

        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        pos_after = pipeline_state.x.pos[0, 2]  # z coordinate of torso
        uph_cost = (pos_after - 0) / self.dt
        quad_ctrl_cost = 0.01 * jp.sum(jp.square(action))
        # quad_impact_cost is not computed here

        obs = self._get_obs(pipeline_state, action)
        reward = uph_cost + 1 - quad_ctrl_cost
        state.metrics.update(reward_linup=uph_cost, reward_quadctrl=-quad_ctrl_cost)

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)
    
    def _get_obs(
        self, pipeline_state: base.State, action: jax.Array
    ) -> jax.Array:
        """Observes humanoid body position, velocities, and angles."""
        position = pipeline_state.q[2:]
        velocity = pipeline_state.qd

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

        # external_contact_forces are excluded
        return jp.concatenate([
            position,
            velocity,
            com_inertia.ravel(),
            com_velocity.ravel(),
            qfrc_actuator,
        ])

    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = (
            jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i