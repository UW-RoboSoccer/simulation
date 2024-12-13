import mujoco
import pickle
from mujoco import mjx

with open("rollout.pkl", "rb") as f:
    rollout, mj_model, mj_data = pickle.load(f)

# Render the rollout
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:    
    while viewer.is_running():
        for frame in rollout:
            mjx.get_data_into(mj_data, mj_model, frame)
            viewer.sync()
            print(mj_data.time)


# Can't do something like mj_data = frame since mj_data is a reference to the memory buffer managed by MuJoCo, so that reassigns the reference but doesn't update the data insde the viewer
