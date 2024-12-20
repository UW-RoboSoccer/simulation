# import numpy as np
# from agents.agent import Agent
# from agents.humanoid import Humanoid
# import six
# from gym import spaces
import os

import brax 
from brax.io import mjcf
from brax import base
from brax import actuator
from brax.envs.base import PipelineEnv, State
from brax import math 

import mujoco
from mujoco import mjx
from mujoco.mjx._src import support  # nit: find a better way to import this (if possible)

import jax
from jax import numpy as jp

POSITION_QUATERNION_SIZE = 7 # = xyz (3) + quaternion (4)
VELOCITY_SIZE = 6 # = linear (3) + angular velocity (3)

class HumanoidKicker(PipelineEnv):
    def __init__(self, **kwargs):
        path = os.path.join(os.path.dirname(__file__), "..", "assets", "kicking", "scene.xml" )
        mj_model = mujoco.MjModel.from_xml_path(str(path))
        self.mj_data = mujoco.MjData(mj_model)
        self.mjx_model = mjx.put_model(mj_model)

        sys = mjcf.load_model(mj_model)
        self.sys = sys


        super().__init__(sys, backend='mjx', **kwargs)


        self.sensorData = {'accel' : [],
                            'gyro' : []}
                            

        self.floor_geom_id = support.name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.left_foot_geom_id = support.name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'leftFoot')
        self.right_foot_geom_id = support.name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'rightFoot')

        print(self.left_foot_geom_id, self.right_foot_geom_id, self.floor_geom_id)


         # #TODO:: Add names to all geoms in op3_simplified.xml for easy indexing
        # # test the force collisions
        # # root body quat and acceleration should be referenced from sensor data from now on
        # # figure out code format convention (gonna go with underscores rather than camelCase)
        # self.links = self.sys.link_names
   
        # print(self.mj_data.ncon)

        # foot_forces=[]
        # for i in range(self.mj_data.ncon):
        #     contact = self.mj_data.contact[i]
        #     # Check if the contact involves the left or right foot
        #     if contact.geom1 == self.lFootid or contact.geom2 == self.lFootid:
        #         force = jp.zeros(6)  # Force-torque vector
        #         mujoco.mj_contactForce(mj_model, self.mj_data, i, force)
        #         foot_forces.append(("left_foot", force))
        #     elif contact.geom1 == self.rFootid or contact.geom2 == self.rFootid:
        #         force = jp.zeros(6)  # Force-torque vector
        #         mujoco.mj_contactForce(mj_model, self.mj_data, i, force)
        #         foot_forces.append(("right_foot", force))
        # print(foot_forces)

        # # Print the contact forces for debugging
        # for foot, force in foot_forces:
        #     print(f"{foot}: Force: {force[:3]}, Torque: {force[3:]}")


        # print('leftFootid:', self.lFootid)
        # print('rightFootid:', self.rFootid)

        # print('data.contact:')
        # print(self.mj_data.contact)


        # print(self.links)

        # num_geoms = mj_model.ngeom
        # print('ngeom: ', num_geoms)
        # geom_list = []
        # for geom_id in range(num_geoms):
        #     geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        #     geom_list.append((geom_id, geom_name))

        # Print the list of geoms
        # for geom_id, geom_name in geom_list:
        #     print(f"Geom ID: {geom_id}, Geom Name: {geom_name}")

        # print(sys.link_names)

        self.humanoid_links = ['torso', 'lwaist', 'pelvis', 'right_thigh', 'right_shin', 'left_thigh', 'left_shin', 'right_upper_arm', 'right_lower_arm', 'left_upper_arm', 'left_lower_arm'] 
        # self.ball_index= self._get_link_index("ball")   
        # self.target_index = self._get_link_index("target")
        # self.humanoid_index = [self._get_link_index(link) for link in self.humanoid_links]
        self.op3_links = [ 'body_link', 'head_pan_link', 'head_tilt_link', 'l_sho_pitch_link', 'l_sho_roll_link', 'l_el_link', 'r_sho_pitch_link', 'r_sho_roll_link', 'r_el_link',
         'l_hip_yaw_link', 'l_hip_roll_link', 'l_hip_pitch_link', 'l_knee_link', 'l_ank_pitch_link', 'l_ank_roll_link', 
         'r_hip_yaw_link', 'r_hip_roll_link', 'r_' ]



        #TODO:: Not hardcode this  
        # print(len(self.sys.init_q))
        # print(self.sys.init_q)
        """
        First 0-2 - xyz for root body 
        Next 3-6 - quaternion for root body 
        5-23 - humanoid dof angles
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


        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=-0.01, maxval=0.01
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=-0.01, maxval=0.01
        )
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
        
        done = jp.asarray(doneStand, dtype=jp.float32) # + doneKick

        # done = doneStand

        #Control cost already calculated in both functions^
        reward = standReward #+ kickReward  #need to linearly decrease one and increase the other.

        # Update metrics
        metrics = {
            'stabilizeReward': standReward,
            'kickReward': 0.0,
        }
        state.metrics.update(metrics)

        # Return updated state
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

<<<<<<< HEAD
    def _get_obs(
        self, pipeline_state: base.State, action: jax.Array
    ) -> jax.Array:
        """Observes humanoid body position, velocities, and angles."""

        # ------- Keep on getting overflow errors can't verify if this is correct! ------ #

        # print('pipeline_state.contacts:', pipeline_state.contacts)

        # contact_forces = jp.zeros((2, 3))

        # for contact in pipeline_state.contacts:
        #     if contact.geom1 == self.lFootid:
        #         contact_forces[0] = contact.force
        #     elif contact.geom1 == self.rFootid:
        #         contact_forces[1] = contact.force

        # print('contact_forces 1:', contact_forces[0])
        # print('contact_forces 2:', contact_forces[1])

        #assume COM pelvis 
        position = pipeline_state.x.pos[2]  #same as below but for position
        velocity = pipeline_state.xd.vel[2] #index 2 is the pelvis. 3 velocity values for every body

        print("Velocity Array: ", velocity)
        print(pipeline_state.xd.vel)

        self.sensorData['accel'].append(self.mj_data.sensor('accel').data.copy())
        self.sensorData['gyro'].append(self.mj_data.sensor('gyro').data.copy())

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
            self.sensorData,
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
