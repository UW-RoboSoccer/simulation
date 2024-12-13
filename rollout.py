import mujoco
import jax
import time

from pathlib import Path
from mujoco import mjx
from jax import numpy as jp

from copy import deepcopy
import pickle

HUMANOID_SCENE_PATH = Path('.') / "assets" / "humanoid" / "humanoid.xml"
OP3_SCENE_PATH = Path('.') / "assets" / "op3" / "scene.xml"
BUILD_SCENE_PATH = Path('.') / "assets" / "build" / "build.xml"

def build_rollout(model_path, n_steps=500):

    # Load the MJ model and data
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    mj_data = mujoco.MjData(mj_model)

    # Save the initial state for rendering purposes
    init_mj_data = deepcopy(mj_data)

    # Map MJ models to MJX to vectorize the data
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)


    print(mjx_data.qpos.devices())

    print('JIT-compiling the step function...')

    # JIT-compile the step function
    jit_step = jax.jit(mjx.step)

    # Create a rollout buffer
    rollout = []
    rng = jax.random.PRNGKey(0)

    print('compilation finished')

    # Rollout simulation and collect rollout data
    for i in range(n_steps):
        if i % 50 == 0:
            print(f'Step {i}')
        ## Avoid something like this, can lead to index-depndent correlations in certain situations |
        ## Can lead to loss of reproducibility in parallel settings                                 v  
        # mj_data.ctrl = jax.random.uniform(jax.random.PRNGKey(i), (mj_data.ctrl.shape[0],), jp.float32, minval=-1, maxval=1)

        # Do the following instead 
        rng, ctrl_rng = jax.random.split(rng)
        random_ctrl = jax.random.uniform(ctrl_rng, (mj_data.ctrl.shape[0],), jp.float32, minval=-1, maxval=1)

        # Update mjx_data with control and current state
        mjx_data = mjx_data.replace(
            ctrl=random_ctrl,
            qpos=jp.array(mj_data.qpos),
            qvel=jp.array(mj_data.qvel),
            act=jp.array(mj_data.act),
            time=jp.array(mj_data.time),
        )

        ## Can be left out for now considering we aren't changing any static parameters
        # mjx_model = mjx_model.tree_replace({
        #     'opt.gravity': mj_model.opt.gravity,
        #     'opt.tolerance': mj_model.opt.tolerance,
        #     'opt.ls_tolerance': mj_model.opt.ls_tolerance,
        #     'opt.timestep': mj_model.opt.timestep,
        # })

        # Perform a physics step
        mjx_data = jit_step(mjx_model, mjx_data)

        # Sync mjx_data back to mj_data
        mjx.get_data_into(mj_data, mj_model, mjx_data)

        # Store the current state in the rollout for rendering
        rollout.append(mjx_data)

    # print('rollout finished')
    # print(rollout[0].time)

    return rollout, mj_model, init_mj_data


if __name__ == "__main__":
    rollout, mj_model, mj_data = build_rollout(OP3_SCENE_PATH, n_steps=500)

    with open("rollout.pkl", "wb") as f:
        pickle.dump((rollout, mj_model, mj_data), f)
    
    print("Rollout saved to rollout.pkl")

