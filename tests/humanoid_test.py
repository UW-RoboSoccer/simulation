from absl.testing import absltest, parameterized
from soccer_env.core.humanoid import Player, PlayerObservables

from dm_control import mjcf
import numpy as np

# class PlayerTest(parameterized.TestCase):
def test_player_pos_to_actuations():
    player = Player()
    player._build()
    return player.mjcf_model
    

if __name__ == '__main__':
    head = test_player_pos_to_actuations()
    print(head)
        # random_joint_pos = np.random.uniform(-1, 1, len(player.actuators))

        # joint_limits = zip(*(actuator.joint.range for actuator in player.actuators))
        # lower, upper = (np.array(limit) for limit in joint_limits)
        # pose = (random_joint_pos * (upper - lower) + upper + lower) / 2
        # actuation = player.pose_to_actuation(pose)

        # np.testing.assert_allclose(actuation, random_joint_pos)

