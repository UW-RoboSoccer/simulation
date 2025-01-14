import jax
from jax import numpy as jp

import mujoco
import mujoco.viewer
from mujoco import mjx

from brax import envs

from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model

import cv2
from PIL import Image

env_name = 'humanoid'
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

rollout = gen_rollout(env, n_steps=500)
images = env.render(rollout, width=640, height=480)

running = True

while running:
    for i, image in enumerate(images):
        # correct color channels
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('image', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            running = False
            break

cv2.destroyAllWindows()

# Output as mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 60, (640, 480))

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
            