from brax import envs
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from envs.humanoid_standup import HumanoidStandup

import jax
from jax import numpy as jp

import mujoco.viewer
from mujoco import mjx
import time

model_path = r'C:\Users\ethan\robocup\soccer_env\tmp\trained_policies\mjx_brax_policy'

envs.register_environment('humanoid-standup', HumanoidStandup)
env = envs.get_environment('humanoid-standup')

params = model.load_params(model_path)
network = ppo_networks.make_ppo_networks(action_size=env.action_size, observation_size=env.observation_size)
make_inference_fn = ppo_networks.make_inference_fn(network)
inference_fn = make_inference_fn(params)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)

state = jit_reset(jax.random.PRNGKey(0))

mj_model = env.sys.mj_model
mj_data = mujoco.MjData(mj_model)
ctrl = jp.zeros(mj_model.nu)

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        act_rng, rnf = jax.random.split(jax.random.PRNGKey(0))
        obs = env._get_obs(state.pipeline_state, ctrl)
        ctrl, _ = jit_inference_fn(obs, act_rng)

        mj_data.ctrl = ctrl
        mujoco.mj_step(mj_model, mj_data)

        env.pipeline_init(jax.numpy.array(mj_data.qpos), jax.numpy.array(mj_data.qvel))

        viewer.sync()
