import jax
from jax import numpy as jp

import mujoco
import mujoco.viewer

import pickle

from brax import envs

from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model

import cv2

from envs.walking import HumanoidWalker
import os

env_name = 'walker_ernest'
# backend = 'positional'
envs.register_environment(env_name, HumanoidWalker)
env = envs.get_environment(env_name=env_name) #, xml_path='assets/humanoid/modified_humanoid.xml')

xml_path='assets/humanoid/ernest_humanoid.xml'
model_path = 'output/params'
# model_path = None
# env = envs.get_environment(env_name)

def gen_rollout(env, model_path=None, n_steps=250):
    jit_inference_fn = None

    if model_path:
        print("model path", model_path)
        print(env.action_size, env.observation_size)
        params = model.load_params(model_path)
        network = ppo_networks.make_ppo_networks(action_size=16, observation_size=env.observation_size)
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
    stats = [state.metrics]
    info = [state.info]

    for i in range(n_steps):
        print('Step', i)
        if jit_inference_fn:
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
        else:
            ctrl = jp.zeros(env.sys.nu)

        state = jit_step(state, ctrl)
        stats.append(state.metrics)
        rollout.append(state.pipeline_state)
        info.append(state.info)

    return rollout, stats, info

rollout, stats, info = gen_rollout(env, n_steps=1000, model_path=model_path)

import matplotlib.pyplot as plt

total_reward = [stat['total_reward'] for stat in stats]
velocity_reward = [stat['velocity_reward'] for stat in stats]
base_height_reward = [stat['base_height_reward'] for stat in stats]
control_cost_reward = [stat['control_cost'] for stat in stats]
rotation_cost = [stat['rotation_cost'] for stat in stats]
upright_rewards = [stat['upright_reward'] for stat in stats]


#Plot Stabilization Metrics
plt.figure(figsize=(12, 8))
plt.plot(total_reward, label='Total Reward')
plt.plot(velocity_reward, label='Velocity Reward')
plt.plot(base_height_reward, label='Base Height Reward')
plt.plot(control_cost_reward, label='Control Cost Reward')
plt.plot(rotation_cost, label='Rotation Cost')
plt.plot(upright_rewards, label='Upright Rewards')
plt.xlabel('Time Step')
plt.ylabel('Reward')
plt.title('Reward Metrics Over Time')
plt.legend()
plt.savefig('output/metrics.png')

# Save rollout to pkl file
with open('output/rollout.pkl', 'wb') as f:
    pickle.dump(rollout, f)
    
if rollout is None:
    with open('output/rollout.pkl', 'rb') as f:
        rollout = pickle.load(f)

images = env.render(rollout, width=640, height=480)

running = True

while running:
    for i, image in enumerate(images):
        # image = cv2.putText(image, f'Height: {info[i]["base_height"]:.2f}', 
        # (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # image = cv2.putText(image, f'Orientation: {info[i]["base_orientation"]}', 
        # (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # correct color channels
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('simulation', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            running = False
            break

cv2.destroyAllWindows()

# Output as mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/output.mp4', fourcc, 60, (640, 480))

for image in images:
    # correct color channels
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out.write(image)

out.release()

# mj_model = env.sys.mj_model
# mj_data = mujoco.MjData(mj_model)
# with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
#     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
#     while viewer.is_running():
#         for i, frame in enumerate(rollout):
#             mj_data.qpos, mj_data.qvel = frame.q, frame.qd
#             mujoco.mj_forward(mj_model, mj_data)

#             print('Humanoid Height:', mj_data.qpos[2])
            
#             # each frame is 0.003s so to show at 60 fps would mean each frame should be shown for 0.0167s so we skip 5 frames
#             if i % 5 == 0:
#                 viewer.sync()
            