import jax
import jax.numpy as jp
from envs.humanoid_kicker import HumanoidKicker

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

# Perform a few steps in the environment
for step in range(5):
    # Generate a random action
    action = jax.random.uniform(rng, (env.sys.act_size(),), minval=-1, maxval=1)
    
    # Step the environment
    state = env.step(state, action)
    
    # Print the state after each step
    print(f"\nState after step {step + 1}:")
    print("Observation:", state.obs)
    print("Reward:", state.reward)
    print("Done:", state.done)
    print("Metrics:", state.metrics)