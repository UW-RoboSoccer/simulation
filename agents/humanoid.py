from .agent import Agent
from gym.spaces import Box
import numpy as np

def mass_center(mass, xpos):
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class Humanoid(Agent):

    def __init__(self, agent_id, xml_path=None, **kwargs):
        if xml_path is None:
            xml_path = "assets/humanoid/humanoid.xml"
        super(Humanoid, self).__init__(agent_id, xml_path, **kwargs)
        self.team='walker'
    
    def set_goal(self, goal):
        self.GOAL = goal
        self.move_left = False
        if self.get_qpos()[0] > 0:
            self.move_left = True
    
    def before_step(self):
        self._pos_before = mass_center(self.get_body_mass(), self.get_xipos())

    def after_step(self):
        raise NotImplementedError
    
    def _get_obs(self):
        raise NotImplementedError
    
    def _get_obs_relative(self):
        raise NotImplementedError
    
    def set_observation_space(self):
        obs = self._get_obs()
        self.obs_dim = obs.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = Box(low, high)

    def reached_goal(self):
        xpos = self.get_body_com('torso')[0]
        if self.GOAL > 0 and xpos > self.GOAL:
            return True
        elif self.GOAL < 0 and xpos < self.GOAL:
            return True
        return False

    def reset_agent(self):
        xpos = self.get_qpos()[0]
        if xpos * self.GOAL > 0 :
            self.set_goal(-self.GOAL)
        if xpos > 0:
            self.move_left = True
        else:
            self.move_left = False