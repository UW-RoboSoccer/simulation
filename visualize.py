import jax
from jax import numpy as jp

import mujoco
import mujoco.viewer
from mujoco import mjx

from brax import envs

from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model, html

from IPython.core.display import HTML

env_name = 'humanoid'
env = envs.get_environment(env_name)

def gen_rollout(env, n_steps=500):
    print('JIT env functions')
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    print('JIT finished')

    state = jit_reset(jax.random.PRNGKey(0))
    rollout = [state.pipeline_state]

    for i in range(n_steps):
        print('Step', i)
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