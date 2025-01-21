from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.io import model

from datetime import datetime

from envs.kicking import HumanoidKick

import functools

import jax

from matplotlib import pyplot as plt

##Feet currently spawns inside the ground

envs.register_environment('kicker', HumanoidKick)
env = envs.get_environment('kicker') #, xml_path='assets/humanoid/humanoid_pos.xml')

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

train_fn = functools.partial(
    ppo.train, num_timesteps=200_000_000, num_evals=5,
    episode_length=1000, normalize_observations=True, action_repeat=1,
    unroll_length=10, num_minibatches=24, num_updates_per_batch=8,
    discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=3072,
    batch_size=512, seed=0
)

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

max_y, min_y = 600, -600
def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    ydataerr.append(metrics['eval/episode_reward_std'])
    print(f'Metrics: {metrics}')

    plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f'y={y_data[-1]:.3f}')

    plt.errorbar(
        x_data, y_data, yerr=ydataerr)
    plt.savefig('output/longtrain.png')

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
model.save_params('output/params', params)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')