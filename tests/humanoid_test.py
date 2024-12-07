from absl.testing import absltest, parameterized
from soccer_env.core.humanoid import Player, PlayerObservables
from soccer_env.core.flatplane import FlatPlane
from soccer_env.core.sphere import Sphere
from soccer_env.core.soccer_ball import SoccerBall
from dm_control.viewer import launch
from dm_control.composer import NullTask
from soccer_env.core.task import SimpleTask
from dm_control import composer
from dm_control.locomotion.walkers.cmu_humanoid import CMUHumanoid

from dm_control import mjcf
import numpy as np

def test_compile_and_step_simulation():
    player = Player()
    player._build()
    return player.actuators

def test_can_compile_and_step_simulation():
    player = Player()
    physics = mjcf.Physics.from_mjcf_model(player.mjcf_model)
    for _ in range(100):
        physics.step()
        print(physics.bind(player.head).xpos)

def test_player_with_null_task():
    """Test that a player can be created without a task in a simple arena."""
    player = Player()
    player._build()
    player._build_observables()
    arena = FlatPlane()
    arena._build()
    soccerball = SoccerBall()
    soccerball._build()
    # arena.add_free_entity(player)
    # soccerball.create_root_joints(arena.attach(sphere)) ## create_root_joints specifies how the player is attached to its parent frame, while arena.attach provides a frame for the entity and adds a freejoint to allow free movement
    task = NullTask(root_entity=player)

    # task = SimpleTask(player, arena)
    env = composer.Environment(task=task, time_limit=45., random_state=None)

    return env

def test_proprioception():
    player = Player()
    player._build()
    head = player.head
    end_effectors = player.end_effectors
    proprioception = player._build_observables().head_height
    return proprioception



if __name__ == '__main__':
    env = test_player_with_null_task()
    launch(env, title="Humanoid Soccer Player in Flat Plane")

    # test_player_with_null_task()
    # head = test_proprioception()
    # print(head)
    # test_can_compile_and_step_simulation()


    # print(test_proprioception())

    # head = test_player_pos_to_actuations()
    # print(head)
    # test_can_compile_and_step_simulation()

    # random_joint_pos = np.random.uniform(-1, 1, len(player.actuators))

    # joint_limits = zip(*(actuator.joint.range for actuator in player.actuators))
    # lower, upper = (np.array(limit) for limit in joint_limits)
    # pose = (random_joint_pos * (upper - lower) + upper + lower) / 2
    # actuation = player.pose_to_actuation(pose)

    # np.testing.assert_allclose(actuation, random_joint_pos)