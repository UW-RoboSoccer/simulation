from absl.testing import absltest, parameterized
from soccer_env.core.humanoid import Player, PlayerObservables

from dm_control import mjcf
import numpy as np

def test_compile_and_step_simulation():
    player = Player()
    physics = mjcf.Physics.from_mjcf_model(player.mjcf_model)
    for _ in range(100):
        physics.step()

def test_player_observables():
    player = Player()
    for observable in player.observables:
        print(observable)

if __name__ == '__main__':
    # test_compile_and_step_simulation()
    test_player_observables()

