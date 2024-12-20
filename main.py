import mujoco
import jax
import time

from pathlib import Path
from mujoco import mjx
from jax import numpy as jp

import mujoco.viewer

HUMANOID_SCENE_PATH = Path('.') / "assets" / "humanoid" / "humanoid_pos.xml"
OP3_SCENE_PATH = Path('.') / "assets" / "kicking" / "scene.xml"

mj_model = mujoco.MjModel.from_xml_path(str(OP3_SCENE_PATH))
mj_data = mujoco.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(mjx_data.qpos.devices())

print('JIT-compiling the step function...')
jit_step = jax.jit(mjx.step)
print('compilation finished')

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        # mj_data.ctrl = jax.random.uniform(jax.random.PRNGKey(time.time_ns()), (mj_data.ctrl.shape[0],), jp.float32, minval=-1, maxval=1)
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