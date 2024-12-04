import colorsys
import os

from absl import logging
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.variation import distributions
from dm_control.entities import props
from dm_control.locomotion.soccer import team
import numpy as np

from dm_control.utils import io as resources 

_ASSETS_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'pitch')

_TOP_CAMERA_Y_PADDING_FACTOR = 1.1
_TOP_CAMERA_DISTANCE = 95.
_GROUND_GEOM_GRID_RATIO = 1. / 100
_FIELD_BOX_CONTACT_BIT = 1 << 7
_SIDE_WIDTH = 32. / 6.
_DEFAULT_PITCH_SIZE = (9, 6)
_DEFAULT_GOAL_LENGTH_RATIO = 2.6 / 6.

_GOALPOST_RELATIVE_SIZE = 0.07
_SUPPORT_POST_RATIO = 0.75

_GOALPOSTS = {'right_post': (1, -1, -1, 1, -1, 1),
              'left_post': (1, 1, -1, 1, 1, 1),
              'top_post': (1, -1, 1, 1, 1, 1),
              'right_base': (1, -1, -1, -1, -1, -1),
              'left_base': (1, 1, -1, -1, 1, -1),
              'back_base': (-1, -1, -1, -1, 1, -1),
              'right_support': (-1, -1, -1, .2, -1, 1),
              'right_top_support': (.2, -1, 1, 1, -1, 1),
              'left_support': (-1, 1, -1, .2, 1, 1),
              'left_top_support': (.2, 1, 1, 1, 1, 1)}

_NUM_HOARDING = 30

def _get_texture(name):
  contents = resources.GetResource(
      os.path.join(_ASSETS_PATH, '{}.png'.format(name)))
  return mjcf.Asset(contents, '.png')

def _top_down_cam_fovy(size, top_camera_distance):
  return (360 / np.pi) * np.arctan2(_TOP_CAMERA_Y_PADDING_FACTOR * max(size),
                                    top_camera_distance)

def _reposition_corner_lights(lights, size):
  """Place four lights at the corner of the pitch."""
  mean_size = 0.5 * sum(size)
  height = mean_size * 2/3
  counter = 0
  for x in [-size[0], size[0]]:
    for y in [-size[1], size[1]]:
      position = np.array((x, y, height))
      direction = -np.array((x, y, height*2))
      lights[counter].pos = position
      lights[counter].dir = direction
      counter += 1

def _goalpost_radius(size):
  """Compute goal post radius as scaled average goal size."""
  return _GOALPOST_RELATIVE_SIZE * sum(size) / 3.


def _post_radius(goalpost_name, goalpost_radius):
  """Compute the radius of a specific goalpost."""
  radius = goalpost_radius
  if 'top' in goalpost_name:
    radius *= 1.01  # Prevent z-fighting at the corners.
  if 'support' in goalpost_name:
    radius *= _SUPPORT_POST_RATIO  # Suport posts are a bit narrower.
  return radius

def _goalpost_fromto(unit_fromto, size, pos, direction):
  """Rotate, scale and translate the `fromto` attribute of a goalpost.

  The goalposts are defined in the unit cube [-1, 1]**3 using MuJoCo fromto
  specifier for capsules, they are then flipped according to whether they face
  in the +x or -x, scaled and moved.

  Args:
    unit_fromto: two concatenated 3-vectors in the unit cube in xyzxyz order.
    size: a 3-vector, scaling of the goal.
    pos: a 3-vector, goal position.
    direction: a 3-vector, either (1,1,1) or (-1,-1,1), direction of the goal
      along the x-axis.

  Returns:
    two concatenated 3-vectors, the `fromto` of a goal geom.
  """
  fromto = np.array(unit_fromto) * np.hstack((direction, direction))
  #why are they using 2size and 2pos? Look into this later
  return fromto*np.array(size+size) + np.array(pos+pos)

class Goal(props.PositionDetector):
  #Goal for soccer + PositionDetector with goalposts

    def _move_goal(self, pos, size):
        for geom in self._goal_geoms:
            unit_fromto = _GOALPOSTS[geom.name]
            geom.fromto = _goalpost_fromto(unit_fromto, size, pos, self._direction)
            geom.size = (_post_radius(geom.name, self._goalpost_radius),)

    def _build(self, direction, net_rgba=(1, 1, 1, .15), make_net=False, **kwargs):
        """Builds the goalposts and net.

        Args:
        direction: Is the goal oriented towards positive or negative x-axis.
        net_rgba: rgba value of the net geoms.
        make_net: Where to add net geoms.
        **kwargs: arguments of PositionDetector superclass, see therein.

        Raises:
        ValueError: If either `pos` or `size` arrays are not of length 3.
        ValueError: If direction in not 1 or -1.
        """
        if len(kwargs['size']) != 3 or len(kwargs['pos']) != 3:
            raise ValueError('`pos` and `size` should be 3-vectors.')
        if direction not in [-1, 1]:
           raise ValueError('`direction` should be either 1 or -1.')
        
        self._direction = np.array((direction, direction, 1))
        
        kwargs['visible'] = False
        super()._build(retain_substep_directions=True, **kwargs)
         
        size = kwargs['size']
        pos = kwargs['pos']
        self._goalpost_radius = _goalpost_radius(size)
        self._goal_geoms = []
        for geom_name, unit_fromto in _GOALPOSTS.items():
            geom_fromto = _goalpost_fromto(unit_fromto, size, pos, self._direction)
            geom_size = (_post_radius(geom_name, self._goalpost_radius),)
            self._goal_geoms.append(
                self._mjcf_root.worldbody.add(
                    'geom',
                    type='capsule',
                    name=geom_name,
                    size=geom_size,
                    fromto=geom_fromto,
                    rgba=self.goalpost_rgba))
            
    def resize(self, pos, size):
        """Call PositionDetector.resize(), move the goal."""
        super().resize(pos, size)
        self._goalpost_radius = _goalpost_radius(size)
        self._move_goal(pos, size)

    def set_position(self, physics, pos):
        """Call PositionDetector.set_position(), move the goal."""
        super().set_position(pos)
        size = 0.5*(self.upper - self.lower)
        self._move_goal(pos, size)

    def _update_detection(self, physics):
        """Call PositionDetector._update_detection(), then recolor the goalposts."""
        super()._update_detection(physics)
        if self._detected and not self._previously_detected:
            physics.bind(self._goal_geoms).rgba = self.goalpost_detected_rgba
        elif self._previously_detected and not self._detected:
            physics.bind(self._goal_geoms).rgba = self.goalpost_rgba
 
    @property
    def goalpost_rgba(self):
        """Goalposts are always opaque."""
        rgba = self._rgba.copy()
        rgba[3] = 1
        return rgba

    @property
    def goalpost_detected_rgba(self):
        """Goalposts are always opaque."""
        detected_rgba = self._detected_rgba.copy()
        detected_rgba[3] = 1
        return detected_rgba

class Pitch(composer.Arena):
    """A pitch with a plane, two goals and a field with position detection."""

    def _build(self,
                size=_DEFAULT_PITCH_SIZE,
                goal_size=None,
                top_camera_distance=_TOP_CAMERA_DISTANCE,
                field_box=False,
                field_box_offset=0.0,
                hoarding_color_scheme_id=0,
                name='pitch'):

        """Construct a pitch with walls and position detectors.
        Args:
        size: a tuple of (length, width) of the pitch.
        goal_size: optional (depth, width, height) indicating the goal size.
            If not specified, the goal size is inferred from pitch size with a fixed
            default ratio.
        top_camera_distance: the distance of the top-down camera to the pitch.
        field_box: adds a "field box" that collides with the ball but not the
            walkers.
        field_box_offset: offset for the fieldbox if used.
        hoarding_color_scheme_id: An integer with value 0, 1, 2, or 3, specifying
            a preset scheme for the hoarding colors.
        name: the name of this arena.
        """
        super()._build(name=name)
        self._size = size
        self._goal_size = goal_size
        self._top_camera_distance = top_camera_distance
        self._hoarding_color_scheme_id = hoarding_color_scheme_id

        self._top_camera = self._mjcf_root.worldbody.add(
            'camera',
            name='top_down',
            pos=[0, 0, top_camera_distance],
            zaxis=[0, 0, 1],
            fovy=_top_down_cam_fovy(self._size, top_camera_distance))

        # Set the `extent`, an "average distance" to 0.1 * pitch length.
        extent = 0.1 * max(self._size)
        self._mjcf_root.statistic.extent = extent
        self._mjcf_root.statistic.center = (0, 0, extent)
        # The near and far clipping planes are scaled by `extent`.
        self._mjcf_root.visual.map.zfar = 5*9             # 5 pitch lengths
        self._mjcf_root.visual.map.znear = 0.1 / extent  # 10 centimeters

        # Add skybox.
        self._mjcf_root.asset.add(
            'texture',
            name='skybox',
            type='skybox',
            builtin='gradient',
            rgb1=(.7, .9, .9),
            rgb2=(.03, .09, .27),
            width=400,
            height=400)

        # Add and position corner lights.
        self._corner_lights = [self._mjcf_root.worldbody.add('light', cutoff=60)
                            for _ in range(4)]
        _reposition_corner_lights(self._corner_lights, size)

        # Increase shadow resolution, (default is 1024).
        self._mjcf_root.visual.quality.shadowsize = 8192

        # Build groundplane.
        if len(self._size) != 2:
            raise ValueError('`size` should be a sequence of length 2: got {!r}'
                            .format(self._size))
        self._field_texture = self._mjcf_root.asset.add(
            'texture',
            type='2d',
            file=_get_texture('pitch'),
            name='fieldplane')
        self._field_material = self._mjcf_root.asset.add(
            'material', name='fieldplane', texture=self._field_texture)

        self._ground_geom = self._mjcf_root.worldbody.add(
            'geom',
            name='ground',
            type='plane',
            material=self._field_material,
            size=list(self._size) + [max(self._size) * _GROUND_GEOM_GRID_RATIO])

        # Build goal position detectors.
        # If field_box is enabled, offset goal by 1.0 such that ball reaches the
        # goal position detector before bouncing off the field_box.
        self._fb_offset = field_box_offset if field_box else 0.0
        goal_size = self._get_goal_size()
        self._home_goal = Goal(
            direction=1,
            make_net=False,
            pos=(-self._size[0] + goal_size[0] + self._fb_offset, 0,
                goal_size[2]),
            size=goal_size,
            rgba=(.2, .2, 1, 0.5),
            visible=True,
            name='home_goal')
        self.attach(self._home_goal)

        self._away_goal = Goal(
            direction=-1,
            make_net=False,
            pos=(self._size[0] - goal_size[0] - self._fb_offset, 0, goal_size[2]),
            size=goal_size,
            rgba=(1, .2, .2, 0.5),
            visible=True,
            name='away_goal')
        self.attach(self._away_goal)

        # Build inverted field position detectors.
        self._field = props.PositionDetector(
            pos=(0, 0),
            size=(self._size[0] - 2 * goal_size[0],
                self._size[1] - 2 * goal_size[0]),
            inverted=True,
            visible=False,
            name='field')
        self.attach(self._field)

        # Build field perimeter.
        def _visual_plane():
            return self._mjcf_root.worldbody.add(
                'geom',
                type='plane',
                size=(1, 1, 1),
                rgba=(0.306, 0.682, 0.223, 1),
                contype=0,
                conaffinity=0)

        self._perimeter = [_visual_plane() for _ in range(8)]
        self._update_perimeter()

        # Build hoarding sites.
        def _box_site():
            return self._mjcf_root.worldbody.add('site', type='box', size=(1, 1, 1))
            self._hoarding = [_box_site() for _ in range(4 * _NUM_HOARDING)]
            self._update_hoarding()

    def _update_hoarding(self):
        # Resize, reposition and re-color visual perimeter box geoms.
        num_boxes = _NUM_HOARDING
        counter = 0
        for dim in [0, 1]:  # Semantics are [x, y]
            width = self._get_goal_size()[2] / 8  # Eighth of the goal height.
            height = self._get_goal_size()[2] / 2  # Half of the goal height.
            length = self._size[dim]
        if dim == 1:  # Stretch the y-dim length in order to cover the corners.
            length += 2 * width
        box_size = height * np.ones(3)
        box_size[dim] = length / num_boxes
        box_size[1-dim] = width
        dim_pos = np.linspace(-length, length, num_boxes, endpoint=False)
        dim_pos += length / num_boxes  # Offset to center.
        for sign in [-1, 1]:
            alt_pos = sign * (self._size[1-dim] * np.ones(num_boxes) + width)
            dim_alt = (dim_pos, alt_pos)
            for box in range(num_boxes):
              box_pos = np.array((dim_alt[dim][box], dim_alt[1-dim][box], width))
            if self._hoarding_color_scheme_id == 0:
                # Red to blue through green + blue hoarding behind blue goal
                angle = np.pi + np.arctan2(box_pos[0], -np.abs(box_pos[1]))
            elif self._hoarding_color_scheme_id == 1:
                # Red to blue through green + blue hoarding behind red goal
                angle = np.arctan2(box_pos[0], np.abs(box_pos[1]))
            elif self._hoarding_color_scheme_id == 2:
                # Red to blue through purple + blue hoarding behind red goal
                angle = np.arctan2(box_pos[0], -np.abs(box_pos[1]))
            elif self._hoarding_color_scheme_id == 3:
                # Red to blue through purple + blue hoarding behind blue goal
                angle = np.pi + np.arctan2(box_pos[0], np.abs(box_pos[1]))
            hue = 0.5 + angle / (2*np.pi)  # In [0, 1]
            hue_offset = .25
            hue = (hue - hue_offset) % 1.0  # Apply offset and wrap back to [0, 1]
            saturation = .7
            value = 1.0
            col_r, col_g, col_b = colorsys.hsv_to_rgb(hue, saturation, value)
            self._hoarding[counter].pos = box_pos
            self._hoarding[counter].size = box_size
            self._hoarding[counter].rgba = (col_r, col_g, col_b, 1.)
            counter += 1

    def _update_perimeter(self):
        # Resize and reposition visual perimeter plane geoms.
        width = self._get_goal_size()[0]
        counter = 0
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if x == 0 and y == 0:
                    continue
                size_0 = self._size[0]-2*width if x == 0 else width
                size_1 = self._size[1]-2*width if y == 0 else width
                size = [size_0, size_1, max(self._size) * _GROUND_GEOM_GRID_RATIO]
                pos = (x*(self._size[0]-width), y*(self._size[1]-width), 0)
                self._perimeter[counter].size = size
                self._perimeter[counter].pos = pos
                counter += 1

    def _get_goal_size(self):
        goal_size = self._goal_size
        if goal_size is None:
            goal_size = (
                _SIDE_WIDTH / 2,
                self._size[1] * _DEFAULT_GOAL_LENGTH_RATIO,
                _SIDE_WIDTH / 2,
        )
        return goal_size

    def register_ball(self, ball):
        self._home_goal.register_entities(ball)
        self._away_goal.register_entities(ball)

        if self._field_box:
        # Geoms a and b collides if:
        #   (a.contype & b.conaffinity) || (b.contype & a.conaffinity) != 0.
        #   See: http://www.mujoco.org/book/computation.html#Collision
            ball.geom.contype = (ball.geom.contype or 1) | _FIELD_BOX_CONTACT_BIT
            for wall in self._field_box:
                wall.conaffinity = _FIELD_BOX_CONTACT_BIT
                wall.contype = _FIELD_BOX_CONTACT_BIT
        else:
            self._field.register_entities(ball)

    def detected_goal(self):
        """Returning the team that scored a goal."""
        if self._home_goal.detected_entities:
            return team.Team.AWAY
        if self._away_goal.detected_entities:
            return team.Team.HOME
        return None

    def detected_off_court(self):
        return self._field.detected_entities

    @property
    def size(self):
        return self._size

    @property
    def home_goal(self):
        return self._home_goal

    @property
    def away_goal(self):
        return self._away_goal

    @property
    def field(self):
        return self._field

    @property
    def ground_geom(self):
        return self._ground_geom

