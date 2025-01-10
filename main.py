import mujoco
import jax
import time

from pathlib import Path
from mujoco import mjx
from jax import numpy as jp

import mujoco.viewer

HUMANOID_SCENE_PATH = Path('.') / "assets" / "humanoid" / "humanoid_pos.xml"
OP3_SCENE_PATH = Path('.') / "assets" / "op3" / "scene.xml"

mj_model = mujoco.MjModel.from_xml_path(str(HUMANOID_SCENE_PATH))
mj_data = mujoco.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# Randomize quaternion orientation of root joint (humanoid only)
# indecies 3:7 are the quaternion values of the root joint, indecies 0:3 are the position values of the root joint
mj_data.qpos[3:7] = jax.random.uniform(jax.random.PRNGKey(time.time_ns()), (4,), jp.float32, minval=-1, maxval=1)
mj_data.qpos[3:7] /= jp.linalg.norm(mj_data.qpos[3:7])  # Normalize to get a valid quaternion

# Randomize base initial height
mj_data.qpos[2] = jax.random.uniform(jax.random.PRNGKey(time.time_ns()), (), jp.float32, minval=0.8, maxval=10)

# Randomize joint positions
mj_data.qpos[7:] = jax.random.uniform(jax.random.PRNGKey(time.time_ns()), (mj_data.qpos[7:].shape[0],), jp.float32, minval=-1, maxval=1)

# Randomize joint velocities
mj_data.qvel = jax.random.uniform(jax.random.PRNGKey(time.time_ns()), (mj_data.qvel.shape[0],), jp.float32, minval=-1, maxval=1)

print(mjx_data.qpos.devices())

print('JIT-compiling the step function...')
jit_step = jax.jit(mjx.step)
print('compilation finished')

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        # mj_data.ctrl = jax.random.uniform(jax.random.PRNGKey(time.time_ns()), (mj_data.ctrl.shape[0],), jp.float32, minval=-1, maxval=1)
        mjx_data = mjx_data.replace(xquat=jp.array(mj_data.xquat))
        mjx_data = mjx_data.replace(ctrl=jp.array(mj_data.ctrl), act=jp.array(mj_data.act))
        mjx_data = mjx_data.replace(qpos=jp.array(mj_data.qpos), qvel=jp.array(mj_data.qvel), time=jp.array(mj_data.time))
        mjx_model = mjx_model.tree_replace({
            'opt.gravity': mj_model.opt.gravity,
            'opt.tolerance': mj_model.opt.tolerance,
            'opt.ls_tolerance': mj_model.opt.ls_tolerance,
            'opt.timestep': mj_model.opt.timestep,
        })

        mjx_data = jit_step(mjx_model, mjx_data)
        mjx.get_data_into(mj_data, mj_model, mjx_data)
        print(mj_data.time)
        viewer.sync()