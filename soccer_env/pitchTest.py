from dm_control import composer
from dm_control.viewer import launch
# from core.pitch import Pitch
from core.pitch import Pitch
from dm_control.composer import NullTask  # Assuming you have access to NullTask

def test_pitch_with_null_task(size=(9, 6), time_limit=45., random_state=None):
    """Test the pitch using NullTask for a simple setup."""
    # Initialize the pitch
    pitch = Pitch()
    pitch._build(size=size, goal_size=(0.6, 2.6, 1.2))

    # Wrap the pitch into a NullTask
    task = NullTask(root_entity=pitch)

    # Create the environment
    env = composer.Environment(
        task=task,
        time_limit=time_limit,
        random_state=random_state
    )

    return env

if __name__ == "__main__":
    # Load the environment
    env = test_pitch_with_null_task()

    # Launch the environment in the viewer for visualization
    launch(env, title="Pitch Test with NullTask")
