import brax
from brax.envs.base import PipelineEnv, State
from brax import envs
from brax.io import mjcf
from brax import base
from brax.base import Transform
from brax import math
import time 

import brax.math
import jax
from jax import numpy as jp

import mujoco
from mujoco import mjx
from mujoco.mjx._src import support

import os

from envs.stand_reward import _calculate_reward

CONTACT_HISTORY_WINDOW = 0.2


class HumanoidKick(PipelineEnv):
    def __init__(self, **kwargs):
        
        path = os.path.join(os.path.dirname(__file__), "..", "assets", "humanoid", "humanoid_pos.xml")
        mj_model = mujoco.MjModel.from_xml_path(str(path))

        self.mj_data = mujoco.MjData(mj_model)
        self.mjx_model = mjx.put_model(mj_model)

        sys = mjcf.load_model(mj_model)
        self.sys = sys

        self.reset_noise_scale = 1e-2

        self.links = self.sys.link_names

        self.seed = jax.random.PRNGKey(0)

        self.accel_end_idx = 3
        self.gyro_end_idx = 6
        self.lin_accel_end_idx = 9

        self.prev_action = jp.zeros(self.sys.act_size())

        super().__init__(self.sys, backend='mjx', **kwargs)

        #IMU Sensor and Contact Force
        self.sensor_data = { 'linear_acceleration': jp.array(3),
                            'angular_velocity' : jp.array(3),
                            'linear_velocity': jp.array(3),
                            'left_contact' : jp.array(1),
                            'right_contact' : jp.array(1)
                            }
        
        #contact forces
        self.floor_geom_id = support.name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.left_foot_geom_id = support.name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'left_foot')
        self.right_foot_geom_id = support.name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'right_foot')
        
        #for foot and ball contacts
        self.ball_geom_id = support.name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'ball')
        self.target_geom_id = support.name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'target') 


    def reset(self, rng: jp.ndarray) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self.reset_noise_scale, self.reset_noise_scale
        humanoid_qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), jp.float32, minval=low, maxval=hi
        )

        humanoid_qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )

        #Reset IMU and Force Sensor Data
        self.sensor_data = { 'linear_acceleration': jp.zeros(3),
                            'angular_velocity' : jp.zeros(3),
                            'linear_velocity': jp.zeros(3),
                            'left_contact' : jp.zeros(1),
                            'right_contact' : jp.zeros(1)
                            }

        # reset ball and target when implemented later

        #concatenate ball and target positions and velocites into this below
        qpos = jp.concatenate([humanoid_qpos])
        qvel = jp.concatenate([humanoid_qvel])

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done = jp.zeros(2)

        metrics = {
            'total_reward': reward,
            'is_standing': 0.0,
            'velocity_reward': 0.0,
            'base_height_reward': 0.0,
            'base_acceleration_reward': 0.0,
            'feet_contact_reward': 0.0,
            'action_diff_reward': 0.0,
            'upright_reward': 0.0
        }

        info = { 
            'random_key': self.seed, 
            'previous_action': jp.zeros(self.sys.act_size())
        }

        # done = True
        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:

        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        obs = self._get_obs(pipeline_state, action)

        is_standing = False
        
        is_standing = jp.where(
            pipeline_state.q[2] > 0.5,
            1.0,
            0.0
        )

        # jax.debug.print("is_standing: {}", is_standing)

        done = jp.where(
            is_standing,
            0.0,
            1.0
        )

        target_height = 1.2

        reward, all_rewards = _calculate_reward(pipeline_state=pipeline_state, sensor_data=self.sensor_data, is_standing=is_standing, 
                    target_height=target_height, action=action, prev_action=action, ctrl_range=self.sys.actuator.ctrl_range)
        # reward = 10
        # left_contact, right_contact = self.get_gait_contact(pipeline_state)

        metrics = {
            'total_reward': reward,
            'is_standing': is_standing,
            'velocity_reward': all_rewards['velocity'],
            'base_height_reward': all_rewards['base_height'],
            'base_acceleration_reward': all_rewards['base_acceleration'],
            'feet_contact_reward': all_rewards['feet_contact'],
            'action_diff_reward': all_rewards['action_difference'],
            'upright_reward': all_rewards['upright']
        }

        # Retrieve the RNG key from the state
        rng = state.info.get('random_seed', self.seed)

        # Debugging: Print the current RNG key
        jax.debug.print("random_seed (before split): {}", rng)

        # Split the RNG key
        rng, next_rng = jax.random.split(rng)

        # Pass the current RNG key to the perturbation function
        pipeline_state = self.add_random_perturbations(pipeline_state, rng)

        # Debugging: Print the updated RNG key
        jax.debug.print("random_seed (after split): {}", next_rng)

        # Save the updated RNG key back into the state
        state_info = {
            'prev_action': action,
            'random_seed': next_rng
        }
        state.info.update(state_info)
        state.metrics.update(metrics)

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        
        position = pipeline_state.q
        velocity = pipeline_state.qd

        data = pipeline_state.sensordata

        self.sensor_data['linear_acceleration'] = data[0: self.accel_end_idx]
        self.sensor_data['angular_velocity'] = data[self.accel_end_idx: self.gyro_end_idx]
        self.sensor_data['linear_velocity'] = data[self.gyro_end_idx: self.lin_accel_end_idx]

        # jax.debug.print("velocimeter data: {}", self.sensor_data['linear_velocity'])
        # jax.debug.print("accel data: {}", self.sensor_data['accel'])
        # jax.debug.print("gyro data: {}", self.sensor_data['gyro'])

        lin_accel = self.sensor_data['linear_acceleration']
        ang_vel = self.sensor_data['angular_velocity']

        #get contact force data
        right_contact_force, left_contact_force = self.get_gait_contact(pipeline_state)
        self.sensor_data['left_contact']= right_contact_force
        self.sensor_data['right_contact'] = left_contact_force        
        # jax.debug.print("right force: {}, left force: {}", self.sensor_data['left_contact'], self.sensor_data['right_contact'])

        # contact_forces = jp.concatenate([right_contact_force, left_contact_force])
        #add actuator values to observation space

        # add target and ball positions

        # Concatenate arrays for output
        return jp.concatenate([
            position, 
            velocity,
            lin_accel,
            ang_vel,
            jp.expand_dims(right_contact_force, axis=0), # expanded for concatenation 
            jp.expand_dims(left_contact_force, axis=0),
            # Add actuators, ball, and target
        ])
    
    def get_gait_contact(self, pipeline_state: base.State):
        # uses support function to model contact forces, see https://github.com/google-deepmind/mujoco/issues/1555
        forces = jp.array([support.contact_force(self.mjx_model, pipeline_state, i) for i in range(pipeline_state.ncon)])

        # checks if, between 2 geoms, there is a point of contact (indicated by True, True)
        right_contacts = pipeline_state.contact.geom == jp.array([self.floor_geom_id, self.right_foot_geom_id])
        left_contacts = pipeline_state.contact.geom == jp.array([self.floor_geom_id, self.left_foot_geom_id])

        # creates a mask based on if there were contacts or not (True, True) == 2 
        right_contact_mask = jp.sum(right_contacts, axis=1) == 2
        left_contact_mask = jp.sum(left_contacts, axis=1) == 2

        # Use masks to filter forces based on contacts 
        total_right_forces = jp.sum(forces * right_contact_mask[:, None], axis=0)
        total_left_forces = jp.sum(forces * left_contact_mask[:, None], axis=0)

        # jax.debug.print("right force: {}, left force: {}", total_right_forces, total_left_forces)

        # returns the y component of the contact forces
        return math.normalize(total_right_forces[:3])[1], math.normalize(total_left_forces[:3])[1]

    
    def get_contact_forces(self, pipeline_state: base.State, forces, geom_id_1, geom_id_2):
        contacts = pipeline_state.contact.geom == jp.array([geom_id_1, geom_id_2])
        contact_mask = jp.sum(contacts, axis=1) == 2
        total_forces = jp.sum(forces * contact_mask[:, None], axis=0)
        return total_forces


    def get_all_contacts(self, pipeline_state: base.State): 
        forces = jp.array([support.contact_force(self.mjx_model, pipeline_state, i) for i in range(pipeline_state.ncon)])

        # 1) Gait contacts
        total_right_gait_forces = self.get_contact_forces(pipeline_state, forces, self.floor_geom_id, self.right_foot_geom_id)
        total_left_gait_forces = self.get_contact_forces(pipeline_state, forces, self.floor_geom_id, self.left_foot_geom_id)

        # 2) Ball contacts
        total_right_ball_forces = self.get_contact_forces(pipeline_state, forces, self.ball_geom_id, self.right_foot_geom_id)
        total_left_ball_forces = self.get_contact_forces(pipeline_state, forces, self.ball_geom_id, self.left_foot_geom_id)

        # 3) Ball and target contacts
        total_ball_target_forces = self.get_contact_forces(pipeline_state, forces, self.ball_geom_id, self.target_geom_id)

        return (
            math.normalize(total_right_gait_forces[:3])[1],
            math.normalize(total_left_gait_forces[:3])[1],
            math.normalize(total_right_ball_forces[:3])[1],
            math.normalize(total_left_ball_forces[:3])[1],
            math.normalize(total_ball_target_forces[:3])[1]
        )
    
    def add_random_perturbations(self, pipeline_state: base.State, rng, frequency=0.2):
        # Split the RNG key to generate independent random values
        rng, rng_force, rng_apply = jax.random.split(rng, 3)

        # Generate a random force
        random_force = jax.random.uniform(rng_force, shape=(3,), minval=-10.0, maxval=10.0)

        # Determine whether to apply the force based on frequency
        apply_force = jax.random.uniform(rng_apply) <= frequency

        # Apply force conditionally
        xfrc_applied = jp.where(
            apply_force,
            pipeline_state.xfrc_applied.at[..., :3].add(random_force),  # Add random force
            pipeline_state.xfrc_applied.at[..., :3].add(jp.zeros(3))    # No force
        )

        # Replace the updated xfrc_applied in the pipeline state
        pipeline_state = pipeline_state.replace(xfrc_applied=xfrc_applied)

        # Debugging: Print the applied force and RNG keys
        jax.debug.print("xfrc_applied: {}, apply_force: {}, random_force: {}", 
                        pipeline_state.xfrc_applied[0][:3], apply_force, random_force)
             
envs.register_environment('kicker', HumanoidKick)

# envs.register_environment('kicker', HumanoidKick)

env = HumanoidKick()
rng = jax.random.PRNGKey(0)

state = env.reset(rng)
obs = env._get_obs(state.pipeline_state, jp.zeros(env.sys.act_size()))

print("Initial State:")
