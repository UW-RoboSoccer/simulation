import mujoco
from brax.io import mjcf
from pathlib import Path
from mujoco import mjx
from envs.humanoid_kicker import HumanoidKicker
import jax 

# Path to your combined XML model
MODEL_PATH = Path('.') / "assets" / "kicking" / "combined.xml"

mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
mj_data = mujoco.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(mjx_data.qpos)
print(mjx_data.qvel)
# # Print the indices of DOFs for all bodies
# print(f"{'Body Name':<20} {'Start Index':<15} {'DOF Count':<10}")
# print("-" * 45)

# # Loop through all bodies in the model
# for body_id in range(sim.model.nbody):
#     body_name = sim.model.id2name(body_id, "body")
#     start_index = sim.model.body_jntadr[body_id]
#     # Find the total DOF count by comparing jnt_dofadr values
#     next_start = (
#         sim.model.body_jntadr[body_id + 1] if body_id + 1 < sim.model.nbody else sim.model.njnt
#     )
#     dof_count = next_start - start_index
#     print(f"{body_name:<20} {start_index:<15} {dof_count:<10}")


# Initialize the environment
env = HumanoidKicker()

# Create a random number generator
rng = jax.random.PRNGKey(0)


# Reset the environment
state = env.reset(rng)

# Print the initial state
print("Initial state:")
print("Observation:", state.obs)
print("Reward:", state.reward)
print("Done:", state.done)
print("Metrics:", state.metrics)

# # Perform a few steps in the environment
# for step in range(5):
#     # Generate a random action
#     action = jax.random.uniform(rng, (env.sys.act_size(),), minval=-1, maxval=1)
    
#     # Step the environment
#     state = env.step(state, action)
    
#     # Print the state after each step
#     print(f"\nState after step {step + 1}:")
#     print("Observation:", state.obs)
#     print("Reward:", state.reward)
#     print("Done:", state.done)
#     print("Metrics:", state.metrics)