import numpy as np
from agents.agent import Agent
from agents.humanoid import Humanoid
import six
from gym import spaces
import os

def mass_center(mass, xpos):
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidKicker(Humanoid):

    def __init__(self, agent_id, xml_path=None):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid", "humanoid.xml" )
