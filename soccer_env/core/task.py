from dm_control import composer

def _disable_geom_contacts(entity):
    mjcf_model = entity.mjcf_model
    for geom in mjcf_model.find_all("geom"):
      geom.set_attributes(contype=0)

class SimpleTask(composer.Task):
    """A simple task that rewards the agent for standing upright."""

    def __init__(self, player, name='simple_task'):
        """Initializes this task.

        Args:
          walker: A `Player` instance.
          control_timestep: A float, the control timestep in seconds.
          name: A string, the name of this task.
        """
        # super().__init__(name=name)
        self._player = player
        # self._control_timestep = control_timestep
        # self.arena = arena

        # _disable_geom_contacts(player)

    @property
    def root_entity(self):
        return self._player
    
    def get_reward(self, physics):
        """Returns a reward to the agent."""
        return 0.0
    
    @property
    def observables(self):
        observables = []
        observables.append(self._player.proprioception)
        return observables
    
    def should_terminate_episode(self, physics):
        return self._player.head_height < 0.5
    
    def before_step(self, physics, actions, random_state):
        self._player.apply_action(physics, actions, random_state)

    


        
    


    # def initialize_episode(self, physics):
    #     """Sets the initial state of the task."""
    #     pass

    # def get_observation(self, physics):
    #     """Returns an observation of the task state."""
    #     return {}

    # def get_reward(self, physics):
    #     """Returns a reward to the agent."""
    #     return 0.0

    # def should_terminate_episode(self, physics):
    #     """Returns a boolean indicating whether the episode should terminate."""
    #     return False

    # def get_discount(self, physics):
    #     """Returns a discount to apply to the reward."""
    #     return 1.0

    # def get_info(self, physics):
    #     """Returns a dictionary of extra information."""
    #     return {}

    # def after_step(self, physics, random_state):
    #     """Runs after each simulation step."""
    #     pass
