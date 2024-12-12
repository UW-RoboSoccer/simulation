# import numpy as np
# from agents.agent import Agent
# from agents.humanoid import Humanoid
# import six
# from gym import spaces
import os

from brax.io import mjcf
from brax import base
from brax import actuator
from brax.envs.base import PipelineEnv, State

import jax
from jax import numpy as jp

# def mass_center(mass, xpos):
#     return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidKicker(PipelineEnv):
    def __init__(self, **kwargs):
        path = os.path.join(os.path.dirname(__file__), "..", "assets", "kicking", "combined.xml" )
        sys = mjcf.load(str(path))

        super().__init__(sys, backend='mjx', **kwargs)

        self.ball_index = self.sys.body.index('soccer_ball')
        self.goal_index = self.sys.body.index('goal')
        self.humanoid_dofs = self.sys.body.index('humanoid')

    def reset(self, rng: jp.ndarray) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # Humanoid's qpos and qvel initialization
        humanoid_qpos = self.sys.init_q[:self.humanoid_dofs] + jax.random.uniform(
            rng1, (self.humanoid_dofs,), minval=-0.01, maxval=0.01
        )
        humanoid_qvel = jax.random.uniform(
            rng2, (self.humanoid_dofs,), minval=-0.01, maxval=0.01
        )

        # Ball and goal use exact XML-defined positions and zero velocities
        ball_qpos = self.sys.init_q[self.ball_dofs_start:self.ball_dofs_end]
        ball_qvel = jp.zeros(self.ball_dofs_end - self.ball_dofs_start)

        goal_qpos = self.sys.init_q[self.goal_dofs_start:self.goal_dofs_end]
        goal_qvel = jp.zeros(self.goal_dofs_end - self.goal_dofs_start)

        # Combine all qpos and qvel
        qpos = jp.concatenate([humanoid_qpos, ball_qpos, goal_qpos])
        qvel = jp.concatenate([humanoid_qvel, ball_qvel, goal_qvel])

        # Initialize pipeline state
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

        goal_pos = pipeline_state.x.pos[self.goal_index]
        ball_pos = self.ball_pos.x.pos[self.ball_index]
        ball_vel = self.ball_vel.xd.vel[self.ball_index]

        # external_contact_forces are excluded
        return jp.concatenate([
            position,
            velocity,
            com_inertia.ravel(),
            com_velocity.ravel(),
            qfrc_actuator,
            goal_pos,
            ball_pos,
            ball_vel,
        ])

    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = (
            jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i


        #  ---------GPT Cooked Code, for reference. However, above code is from ---------  #
            #  ---------humanoidStandup class thus is not correct for this ---------  #
            
# def step(self, state: State, action: jp.ndarray) -> State:
#     # Scale action to actuator range
#     action_min = self.sys.actuator.ctrl_range[:, 0]
#     action_max = self.sys.actuator.ctrl_range[:, 1]
#     action = (action + 1) * (action_max - action_min) * 0.5 + action_min

#     # Simulate physics
#     pipeline_state = self.pipeline_step(state.pipeline_state, action)

#     # Extract positions
#     humanoid_pos = pipeline_state.x.pos[0]  # Humanoid
#     ball_pos = pipeline_state.x.pos[1]  # Ball
#     goal_pos = pipeline_state.x.pos[2]  # Goal

#     # Reward: Encourage interaction with the ball and scoring
#     ball_to_goal_dist = jp.norm(ball_pos - goal_pos)  # Distance from ball to goal
#     humanoid_to_ball_dist = jp.norm(humanoid_pos - ball_pos)  # Distance from humanoid to ball
#     ball_speed = jp.norm(pipeline_state.xd.vel[1])  # Ball velocity

#     reward = -humanoid_to_ball_dist  # Reward approaching the ball
#     reward += jp.exp(-ball_to_goal_dist)  # Bonus for moving ball closer to goal
#     reward += 10.0 if ball_to_goal_dist < 0.1 else 0.0  # Big reward for scoring

#     # Penalty for control effort
#     ctrl_cost = 0.01 * jp.sum(jp.square(action))
#     reward -= ctrl_cost

#     # Termination condition: Check if goal is scored
#     done = jp.array(1.0) if ball_to_goal_dist < 0.1 else jp.array(0.0)

#     # Compute new observations
#     obs = self._get_obs(pipeline_state, action)

#     # Update metrics
#     state.metrics.update(reward_ctrl=-ctrl_cost, reward_goal=reward)

#     # Return updated state
#     return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
