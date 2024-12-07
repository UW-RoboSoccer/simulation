import jax
import numpy as np
from jax import numpy as jnp
from core.humanoid import Humanoid

def test_humanoid():
    rng = jax.random.PRNGKey(42)

    env = Humanoid(
        forward_reward_w=1.0,
        alive_reward=1.0,
        ctrl_cost_w=0.1,
        terminate_when_dead=True,
        alive_z_range=(-0.1, 0.5),
        reset_noise_scale=0.1,
    )

    state = env.reset(rng)
    print("Initial Observation: ", state.obs)

    for step in range(10):
        rng, sub_rng = jax.random.split(rng)
        action = jax.random.uniform(sub_rng, shape=(env.action_size,), minval=-1, maxval=1 )

        state = env.step(state, action)

        print("Observation: ", state.obs)
        print("Reward: ", state.reward)
        print("Done: ", state.done)
        print("Metrics: ", state.metrics)

if __name__ == "__main__":
    test_humanoid()