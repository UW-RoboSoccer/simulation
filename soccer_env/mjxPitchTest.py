from mujoco import mjx
import matplotlib.pyplot as plt
import mujoco
from mujoco import viewer
import jax

xml_path = "./xml_save/pitch.xml"

mj_model = mujoco.MjModel.from_xml_path(xml_path)
mjx_model = mjx.put_model(mj_model)

mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

mjx_data = mjx.make_data(mjx_model)

viewer.launch(mjx_model, mjx_data)
