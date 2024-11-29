import numpy as np
import mujoco as mj

# def _get_texture(name):
#     contents = resources.GetResource(
#         os.path.join(_ASSETS_PATH, '{}.png'.format(name)))
#     return m
#     )

_WALL_HEIGHT = 10
_WALL_THICKNESS = .5
_GOALPOST_REL_SIZE = 0.07 #Ratio of goalpost radius to goal size
_SUPPORT_POST_RATIO = 0.75 #Ratio of support post radius to goalpost radius
def _goalpost_radius(size):
    return _GOALPOST_REL_SIZE * sum(size) / 3.

def _post_radius(goalpost_name, goalpost_radius):
    radius = goalpost_radius
    if 'top' in goalpost_name:
        radius *= 1.01
    if 'support' in goalpost_name:
        radius *= _SUPPORT_POST_RATIO
    return radius


#class Goal(__):

# class SoccerPitch:
#     def __init__(self, total_length = 10, total_width = 7, pitch_length = 9, pitch_width = 6, goal_size=(1.2,2.6)):
#         self.total_length = total_length
#         self.total_width = total_width
#         self.pitch_length = pitch_length
#         self.pitch_width = pitch_width
#         self.goal_size = goal_size
#         self.elements = {"goals": [], "walls": [], "lights": []}

#         self.model = None
#         self.sim = None
#         self.viewer = None

#         self._build_pitch()

    # def _build_pitch(self):
