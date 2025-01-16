import jax
from jax import numpy as jp

import mujoco
import mujoco.viewer
from mujoco import mjx

from brax import envs

from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model, html

from env.op3_runner import OP3Runner

from IPython.core.display import HTML

# env_name = 'humanoid'
env_name = 'op3_runner'
env = envs.get_environment(env_name)

def gen_rollout(env, model_path=None, n_steps=500):
    jit_inference_fn = None

    if model_path:
        params = model.load_params(model_path)
        network = ppo_networks.make_ppo_networks(action_size=env.action_size, observation_size=env.observation_size)
        make_inference_fn = ppo_networks.make_inference_fn(network)
        inference_fn = make_inference_fn(params)
        jit_inference_fn = jax.jit(inference_fn)

    print('JIT env functions')
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    print('JIT finished')

    rng = jax.random.PRNGKey(0)

    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    for i in range(n_steps):
        print('Step', i)
        if jit_inference_fn:
            act_rng, rng = jax.random.split(rng)
            obs = env._get_obs(state.pipeline_state)
            ctrl, _ = jit_inference_fn(obs, act_rng)
        else:
            ctrl = jp.zeros(env.sys.nu)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)

    return rollout

rollout = gen_rollout(env)

mj_model = env.sys.mj_model
mj_data = mujoco.MjData(mj_model)
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:    
    while viewer.is_running():
        for frame in rollout:
            mj_data.qpos, mj_data.qvel = frame.q, frame.qd
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            print(mj_data.time)