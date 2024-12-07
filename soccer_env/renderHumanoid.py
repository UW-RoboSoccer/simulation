import jax
from jax import numpy as jp
import mujoco
from brax import envs

import cv2

from core.humanoid import Humanoid



# # # More legible printing from numpy.

def main(): 
    envs.register_environment('custom_humanoid', Humanoid)
    env = envs.get_environment('custom_humanoid')

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state = jit_reset(jax.random.key(0))
    rollout = [state.pipeline_state]

    # grab a trajectory
    for _ in range(100):
        ctrl = -0.1 * jp.ones(env.sys.nu)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        
    frames = env.render(rollout, camera='side')

    # Display each frame using OpenCV
    for frame in frames:
        cv2.imshow('Video', frame)
        if cv2.waitKey(int(env.dt * 1000)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()