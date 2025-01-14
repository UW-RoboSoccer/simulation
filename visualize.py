import jax
from jax import numpy as jp

import mujoco
import mujoco.viewer
from mujoco import mjx

from brax import envs

from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model, html

from IPython.core.display import HTML

from envs.kicking import HumanoidKick
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

env_name = 'kicker'
env = envs.get_environment(env_name)

os.environ['MUJOCO_GL'] = 'egl'
xla_flags =os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

def gen_rollout(env, model_path=None, n_steps=100):
    
    jit_inference_fn = None

    if model_path:
        params = model.load_params(model_path)
        network = ppo_networks.make_ppo_networks(action_size=env.action_size, observation_size=env.observation_size)
        make_inference_fn = ppo_networks.make_inference_fn(network)
        inference_fn = make_inference_fn(params)
        jit_inference_fn = jax.jit(inference_fn)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng) 

    rollout = [state.pipeline_state]

    metrics = {'step': [], 'reward': [], 'done': [], 'stabilizeReward': [], 'kickReward': [], 
               'gyroscope': [], 'accelerometer': [], 'contactForces': []}



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

        metrics['step'].append(i)
        metrics['reward'].append(float(state.reward))
        metrics['done'].append(bool(state.done))
        metrics['stabilizeReward'].append(float(state.metrics['stabilizeReward']))
        metrics['kickReward'].append(float(state.metrics['kickReward']))
        metrics['gyroscope'].append(state.metrics['gyroscope'])
        metrics['accelerometer'].append(state.metrics['accelerometer'])
        metrics['contactForces'].append(state.metrics['contactForces'])

    metrics_df = pd.DataFrame(metrics)

    return rollout, metrics_df

rollout, metrics_df = gen_rollout(env)

mj_model = env.sys.mj_model
mj_data = mujoco.MjData(mj_model)
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:    
    while viewer.is_running():
        for frame in rollout:
            mj_data.qpos, mj_data.qvel = frame.q, frame.qd
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

output_dir = 'plots'
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=metrics_df, x='step', y='reward', label='Total Reward')
# plt.title('Reward Over Time')
# plt.xlabel('Steps')
# plt.ylabel('Reward')
# plt.legend()
# plt.savefig(f'{output_dir}/total_reward.png')
# plt.close()

# Plotting individual reward components
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=metrics_df, x='step', y='stabilizeReward', label='Stabilize Reward')
# sns.lineplot(data=metrics_df, x='step', y='kickReward', label='Kick Reward')
# plt.title('Reward Components Over Time')
# plt.xlabel('Steps')
# plt.ylabel('Reward')
# plt.savefig(f'{output_dir}/reward_components.png')
# plt.close()

# Plotting done status over time
# plt.figure(figsize=(12, 6))
# sns.scatterplot(data=metrics_df, x='step', y='done', label='Done Status', marker='o', color='red')
# plt.title('Done Status Over Time')
# plt.xlabel('Steps')
# plt.ylabel('Done (1 = True, 0 = False)')
# plt.legend()
# plt.savefig(f'{output_dir}/done_status.png')
# plt.close()

# Plot correlation heatmap for metrics
# plt.figure(figsize=(10, 8))
# sns.heatmap(metrics_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Between Metrics')
# plt.savefig(f'{output_dir}/correlation_heatmap.png')
# plt.close()

plt.figure(figsize=(12, 6))
contact_data = metrics_df['contactForces'].apply(lambda x: pd.Series(x))
sns.lineplot(data=contact_data, label='Right Foot')
sns.lineplot(data=contact_data, label='Left Foot')
plt.title('Contact Forces')
plt.xlabel('Steps')
plt.ylabel('Force (N)')
plt.savefig(f'{output_dir}/contact_forces.png')
plt.close()

