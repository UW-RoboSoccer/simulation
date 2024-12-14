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
from brax import math 

import jax
from jax import numpy as jp

POSITION_QUATERNION_SIZE = 7 # = xyz (3) + quaternion (4)
VELOCITY_SIZE = 6 # = linear (3) + angular velocity (3)


def stabilization_reward(pipeline_state: base.State, action: jp.ndarray, obs, target_height=1.0, dt=0.05) -> float:
    pelvis_pos = obs[0:3]   # Humanoid

    torso_height = pelvis_pos[2]
    height_error = abs(target_height - torso_height)
    height_reward = -height_error / dt

    control_cost = 0.01 * jp.sum(jp.square(action))

    reward = height_reward + 1 - control_cost

    fallen_threshold = 0.5
    done = torso_height < fallen_threshold
    
    return reward, done

def calculate_kick_reward(pipeline_state: base.State, action: jp.ndarray, obs):

    pelvis_pos = obs[0:3]  # Humanoid
    ball_pos = obs[9:12] # Ball
    goal_pos = obs[6:9]  # Goal
    ball_vel = obs[12:15] # Ball velocity

    ball_to_goal_dist = jp.linalg.norm(ball_pos - goal_pos)  # Distance from ball to goal
    humanoid_to_ball_dist = jp.linalg.norm(pelvis_pos - ball_pos)  # Distance from humanoid to ball
   
    ball_speed = jp.linalg.norm(ball_vel)  # Speed of the ball
    reward = -humanoid_to_ball_dist  # Reward approaching the ball
    reward += jp.exp(-ball_to_goal_dist)  # Bonus for moving ball closer to goal
    reward += jp.where(ball_to_goal_dist < 0.1, 10.0, 0.0) 
    reward += 0.1 * ball_speed  # Bonus for kicking the ball    

    ctrl_cost = 0.01 * jp.sum(jp.square(action))
    reward -= ctrl_cost

    done = jp.where(ball_to_goal_dist < 0.1, jp.array(1.0), jp.array(0.0))

    return reward, done

class HumanoidKicker(PipelineEnv):
    def __init__(self, **kwargs):
        path = os.path.join(os.path.dirname(__file__), "..", "assets", "kicking", "combined.xml" )
        sys = mjcf.load(str(path))

        super().__init__(sys, backend='mjx', **kwargs)

        self.humanoid_links = ['torso', 'lwaist', 'pelvis', 'right_thigh', 'right_shin', 'left_thigh', 'left_shin', 'right_upper_arm', 'right_lower_arm', 'left_upper_arm', 'left_lower_arm'] 
        self.ball_index= self._get_link_index("ball")   
        self.target_index = self._get_link_index("target")
        self.humanoid_index = [self._get_link_index(link) for link in self.humanoid_links]


        #TODO:: Not hardcode this  
        """
        print(len(self.sys.init_q))
        First 0-2 - xyz for root body 
        Next 3-6 - quaternion for root body 
        5-23 - humanoid body angles? 
        24-26 - ball xyz
        27-30 ball quaternion 
        31-33 - goal xyz
        34-37 - goal quaternion
        """
        self.ball_pos_start = 24 # Ball position in qpos
        self.target_pos_start = 31 # Target position in qpos
 

    def _get_link_index(self, link_name):
        try:
            return self.sys.link_names.index(link_name)
        except ValueError:
            raise ValueError(f"Link '{link_name}' not found in the system.")


    def reset(self, rng: jp.ndarray) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # Humanoid's qpos and qvel initialization
        humanoid_qpos = self.sys.init_q[:self.ball_pos_start] + jax.random.uniform(
            rng1, (self.ball_pos_start,), minval=-0.01, maxval=0.01
        )
        
        # remove 1 from ball pos start idx (no quaternion)
        humanoid_qvel = jax.random.uniform(
            rng2, (self.ball_pos_start - 1,), minval=-0.01, maxval=0.01
        )

        # Ball and goal use exact XML-defined positions, quaternions and zero velocities
        ball_qpos = self.sys.init_q[self.ball_pos_start: self.ball_pos_start + POSITION_QUATERNION_SIZE] 
        ball_qvel = jp.zeros(VELOCITY_SIZE) # no rot/angular velocity for ball

        target_qpos = self.sys.init_q[self.target_pos_start: self.target_pos_start +POSITION_QUATERNION_SIZE]
        target_qvel = jp.zeros(VELOCITY_SIZE) #  no rot/angular velocity for target

        # Combine all qpos and qvel
        qpos = jp.concatenate([humanoid_qpos, ball_qpos, target_qpos])
        qvel = jp.concatenate([humanoid_qvel, ball_qvel, target_qvel])

        # Initialize pipeline state
        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)
        metrics = {
            'stabilizeReward': zero,
            'kickReward': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    
    def step(self, state: State, action: jp.ndarray) -> State:
        # Scale action to actuator range
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        # Simulate physics
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        # Compute new observations
        obs = self._get_obs(pipeline_state, action)

        # Extract positions

        # Calculate reward
<<<<<<< HEAD
        kickReward, done = calculate_kick_reward(pipeline_state, action, obs)
        standReward = stabilization_reward(pipeline_state, action, obs)
=======
        kickReward, doneKick = calculate_kick_reward(pipeline_state, action)
        standReward, doneStand = stabilization_reward(pipeline_state, action)
>>>>>>> 19a86dc (saving)
        
        done = doneKick + doneStand

        #Control cost already calculated in both functions^
        reward = kickReward + standReward #need to linearly decrease one and increase the other.

     

        # Update metrics
        metrics = {
            'stabilizeReward': standReward,
            'kickReward': kickReward,
        }
        state.metrics.update(metrics)

        # Return updated state
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

<<<<<<< HEAD
    def _get_obs(
        self, pipeline_state: base.State, action: jax.Array
    ) -> jax.Array:
        """Observes humanoid body position, velocities, and angles."""

        #assume COM pelvis 
        position = pipeline_state.x.pos[2] 
        velocity = pipeline_state.xd.vel[2]


        # com, inertia, mass_sum, x_i = self._com_humanoid(pipeline_state)
        # cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        # com_inertia = jp.hstack(
        #     [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[: len(self.humanoid_links)]]
        # )

        # xd_i = (
        #     base.Transform.create(pos=x_i.pos - pipeline_state.x.pos)
        #     .vmap()
        #     .do(pipeline_state.xd)
        # )

        # com_vel = inertia.mass[:len(self.humanoid_links)] * xd_i.vel[:len(self.humanoid_links)] / mass_sum
        # com_ang = xd_i.ang[:len(self.humanoid_links)]
        # com_velocity = jp.hstack([com_vel, com_ang])

        # qfrc_actuator = actuator.to_tau(
        #     self.sys, action, pipeline_state.q, pipeline_state.qd
        # )

        goal_pos = pipeline_state.x.pos[self.target_index]
        ball_pos = pipeline_state.x.pos[self.ball_index]
        ball_vel = pipeline_state.xd.vel[self.ball_index]

        # external_contact_forces are excluded
        return jp.concatenate([
            position,
            velocity,
            # com_inertia.ravel(),
            # com_velocity.ravel(),
            # qfrc_actuator,
            goal_pos,
            ball_pos,
            ball_vel,
        ])

    # def _com(self, pipeline_state: base.State) -> jax.Array:
    #     inertia = self.sys.link.inertia
    #     mass_sum = jp.sum(inertia.mass)
    #     x_i = pipeline_state.x.vmap().do(inertia.transform)
    #     com = (
    #         jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
    #     )
    #     return com, inertia, mass_sum, x_i
=======
    def _get_obs(self, pipeline_state: base.State):
        """Get complete state observation for humanoid kicker environment."""
        # Get center of mass data using class method
        com_pos, com_vel = self._com(pipeline_state)
        
        # Core body state
        qpos = pipeline_state.qp.pos
        qvel = pipeline_state.qp.vel
        cinr = pipeline_state.qp.cinr
        
        # Torso state (root body)
        torso_pos = qpos[0, :3]
        torso_rot = qpos[0, 3:7]
        torso_vel = qvel[0, :3]
        ang_vel = qvel[0, 3:6]
        
        # Joint and actuator states
        joint_pos = qpos[1:]
        joint_vel = qvel[1:]
        qfrc_actuator = pipeline_state.qf.qfrc_actuator
        
        # Relative body positions to COM
        xd_i = qpos[:, :3] - com_pos[None]
        
        # Ball state (second to last body)
        ball_pos = pipeline_state.qp.pos[-2, :3]
        ball_vel = pipeline_state.qp.vel[-2, :3]
        
        # Target state (last body)
        target_pos = pipeline_state.qp.pos[-1, :3]
        
        obs = jp.concatenate([
            com_pos,           # 3 - COM position from _com()
            com_vel,          # 3 - COM velocity from _com()
            cinr.flatten(),   # 9 - Centroidal inertia
            torso_pos,        # 3 - Torso position
            torso_rot,        # 4 - Torso rotation
            torso_vel,        # 3 - Torso linear velocity
            ang_vel,          # 3 - Torso angular velocity
            joint_pos,        # n - Joint positions
            joint_vel,        # n - Joint velocities
            qfrc_actuator,    # n - Actuator forces
            xd_i.flatten(),   # 3*n_bodies - Body positions relative to COM
            ball_pos,         # 3 - Ball position
            ball_vel,         # 3 - Ball velocity
            target_pos,       # 3 - Target position
        ])
        
        return obs
    
    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = (
            jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i
    
>>>>>>> 19a86dc (saving)
