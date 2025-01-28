import jax
from jax import numpy as jp

import mujoco
import mujoco.viewer
from mujoco import mjx

import os
import pickle

from brax import envs

from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model

import cv2

from envs.base_op3 import OP3Stand

envs.register_environment('op3-stand', OP3Stand)

env_name = 'op3-stand'
env = envs.get_environment(env_name, xml_path='assets/humanoid/humanoid_pos.xml')
# env = envs.get_environment('humanoid')

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

rollout, stats, info = gen_rollout(env, model_path='output/params', n_steps=2000)

# Plot reward metrics
import matplotlib.pyplot as plt

# Extract metrics
reward_base_orient = [stat['reward_base_orient'] for stat in stats]
reward_height = [stat['reward_height'] for stat in stats]
quad_ctrl_cost = [stat['reward_ctrl'] for stat in stats]
reward_up_vel = [stat['reward_up_vel'] for stat in stats]

# Plot metrics
plt.figure(figsize=(12, 8))
plt.plot(reward_base_orient, label='Reward Base Orient')
plt.plot(reward_height, label='Reward Height')
plt.plot(quad_ctrl_cost, label='Quad Ctrl Cost')
plt.plot(reward_up_vel, label='Reward Up Vel')
plt.xlabel('Time Step')
plt.ylabel('Reward')
plt.title('Reward Metrics Over Time')
plt.legend()
plt.show()

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

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
        # write out info to text on image
        image = cv2.putText(image, f'Height: {info[i]["base_height"]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        image = cv2.putText(image, f'Orientation: {info[i]["base_orientation"]}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

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
            