from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model
from brax.training import networks

import datetime

from envs.humanoid_standup import HumanoidStandup
from envs.humanoid_kicker import HumanoidKicker

import functools

import jax
import flax.linen as nn
from jax import numpy as jp
import distrax

from matplotlib import pyplot as plt

# class LSTMPolicyValueNetwork(nn.Module):
#     action_size: int
#     action_min: float
#     action_max: float
#     hidden_size: int = 128
#     kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.lecun_uniform()
#     bias: bool = True 

#     @nn.compact
#     def create_lstm_model(self, x, key):
#         """LSTM Model based on paper"""

#         # Pre-LSTM projection (128 from paper)
#         x = nn.Dense(
#             features=self.hidden_size,
#             kernel_init=self.kernel_init,
#             use_bias=self.bias,
#             )(x)
        
#         x = nn.relu(x)
        
#         # LSTM processing
#         lstm_cell = nn.LSTMCell(
#             use_bias=self.bias,

#         )
#         batch_size = x.shape[0]
#         initial_carry = lstm_cell.initialize_carry(
#             jax.random.PRNGKey(key), 
#             (batch_size, self.hidden_size)
#         )
#         x, _ = lstm_cell(initial_carry, x)
        
#         # Post-LSTM projection
#         x = nn.Dense(
#             features=self.hidden_size,
#             kernel_init=self.kernel_init,
#             use_bias=self.bias,
#             )(x)
        
#         x = nn.relu(x)
        
#         return x
    
#     @nn.compact
#     def __call__(self, x, deterministic=False):
#         # Use embedding function with key offset for policy
#         policy_x = self.create_lstm_model(x, self.hidden_size, key_offset=0)
        
#         # Policy mean
#         policy_mean = nn.Dense(features=self.action_size)(policy_x)
        
#         # Learnable log standard deviation for diagonal covariance
#         log_std = self.param(
#             'policy_log_std', 
#             nn.initializers.zeros, 
#             (self.action_size,)
#         )
        
#         # Create Gaussian distribution with clipped outputs
#         policy_std = jp.exp(log_std)
#         policy_dist = distrax.TruncatedNormal(
#             loc=policy_mean, 
#             scale=policy_std,
#             low=self.action_min,
#             high=self.action_max
#         )
        
#         # Value function 
#         value_x = self.create_lstm_model(x, self.hidden_size, key_offset=1)
        
#         # Value function output
#         value = nn.Dense(features=1)(value_x)
        
#         return policy_dist, value.squeeze(-1)

# def create_lstm_policy_value_network(observation_size, action_size, action_min, action_max):
#     def network_factory(num_action_repeats):
#         def policy_network(obs_size, action_size, unused_preprocessing_layer=None):
#             def init(key):
#                 def make_policy_network():
#                     return LSTMPolicyValueNetwork(
#                         action_size=action_size, 
#                         action_min=action_min, 
#                         action_max=action_max
#                     )
                
#                 network = make_policy_network()
#                 return network.init(key, jp.zeros((1, obs_size)))
            
#             def apply(params, obs, key=None):
#                 network = LSTMPolicyValueNetwork(
#                     action_size=action_size, 
#                     action_min=action_min, 
#                     action_max=action_max
#                 )
#                 policy_dist, value = network.apply(params, obs)
#                 return policy_dist, value
            
#             return init, apply
        
#         return policy_network
    
#     return network_factory

envs.register_environment('humanoid-kicker', HumanoidKicker)
env = envs.get_environment('humanoid-kicker')

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

train_fn = functools.partial(
    ppo.train,
    num_timesteps=20_000_000,
    num_evals=5,
    reward_scaling=0.1,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,  # BPTT 
    num_minibatches=80, 
    num_updates_per_batch=3,  # Paper specified 
    discounting=0.995, # Gamma 
    learning_rate=0.001,
    entropy_cost=0.0, # Paper didn't implement entropy cost 
    num_envs=4096, # 4096 envs
    batch_size=5120, # 5120 samples
    seed=0,
    clipping_epsilon=0.2, # PPO eplison 
    gae_lambda=0.95, # generalized advantage estimate param
)

x_data = []
y_data = []
ydataerr = []
times = [datetime.datetime.now()]

max_y, min_y = 13000, 0
def progress(num_steps, metrics):
    times.append(datetime.datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    ydataerr.append(metrics['eval/episode_reward_std'])

    plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f'y={y_data[-1]:.3f}')

    plt.errorbar(
        x_data, y_data, yerr=ydataerr)
    plt.show()

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')