import os
import numpy as np

from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion.walkers import legacy_base
from dm_control.composer.observation import observable

## Define these later based on physical build
_STAND_HEIGHT = 1.0
_TORQUE_LIMIT = 60.0

_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'humanoid_{model_version}.xml')

class Player(legacy_base.Walker):
    """A humanoid soccer player."""

    def _build(self, name='walker', initializer=None):
        super()._build(initializer)

        self._mjcf_root = mjcf.from_path(self._xml_path)
        if name:
            self._mjcf_root.model = name

        limits = zip(*(actuator.joint.range for actuator in self.actuators))
        lower, upper = (np.array(limit) for limit in limits)
        self._scale = upper - lower
        self._offset = upper + lower

    def _build_observables(self):
        return PlayerObservables(self)
    
    def pose_to_actuation(self, targt_pose):
        return (2 * targt_pose - self._offset) / self._scale

    @property
    def _xml_path(self):
        return _XML_PATH.format(model_version='v0')
    
    @composer.cached_property
    def actuators(self):
        return tuple(self._mjcf_root.find_all('actuator'))

    @composer.cached_property
    def root_body(self):
        return self._mjcf_root.find('body', 'root')

    @composer.cached_property
    def head(self):
        return self._mjcf_root.find('body', 'head')

    @composer.cached_property
    def left_arm_root(self):
        return self._mjcf_root.find('body', 'lclavicle')

    @composer.cached_property
    def right_arm_root(self):
        return self._mjcf_root.find('body', 'rclavicle')

    @composer.cached_property
    def ground_contact_geoms(self):
        return tuple(self._mjcf_root.find('body', 'lfoot').find_all('geom') +
                    self._mjcf_root.find('body', 'rfoot').find_all('geom'))

    @composer.cached_property
    def standing_height(self):
        return _STAND_HEIGHT

    @composer.cached_property
    def end_effectors(self):
        return (self._mjcf_root.find('body', 'rradius'),
                self._mjcf_root.find('body', 'lradius'),
                self._mjcf_root.find('body', 'rfoot'),
                self._mjcf_root.find('body', 'lfoot'))

    @composer.cached_property
    def observable_joints(self):
        return tuple(actuator.joint for actuator in self.actuators
                    if actuator.joint is not None)

    @composer.cached_property
    def bodies(self):
        return tuple(self._mjcf_root.find_all('body'))

    @composer.cached_property
    def mocap_tracking_bodies(self):
        """Collection of bodies for mocap tracking."""
        # remove root body
        root_body = self._mjcf_root.find('body', 'root')
        return tuple(
            b for b in self._mjcf_root.find_all('body') if b != root_body)

    @composer.cached_property
    def egocentric_camera(self):
        return self._mjcf_root.find('camera', 'egocentric')

    @composer.cached_property
    def body_camera(self):
        return self._mjcf_root.find('camera', 'bodycam')

class PlayerObservables(legacy_base.WalkerObservables):
    """Observables for a humanoid soccer player."""
    
    @composer.observable
    def head_height(self):
        observable.MJCFFeature('xpos', self._entity.head)[2]

    @composer.observable
    def sensors_torque(self):
        return observable.MJCFFeature(
            'sensordata', self._entity.mjcf_model.sensor.torque,
            corruptor=lambda v, random_state: np.tanh(2 * v / _TORQUE_LIMIT))

    @composer.observable
    def actuator_activation(self):
        return observable.MJCFFeature('act', self._entity.find_all('actuator'))

    @composer.observable
    def appedages_pos(self):
        """Equivalent to `end_effectors_pos` with the head's position appended."""
        def relative_pos_in_egocentric_frame(physics):
            end_effectors_with_head = (
                self._entity.end_effectors + (self._entity.head,))
            end_effector = physics.bind(end_effectors_with_head).xpos
            torso = physics.bind(self._entity.root_body).xpos
            xmat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
            return np.reshape(np.dot(end_effector - torso, xmat), -1)
        return observable.Generic(relative_pos_in_egocentric_frame)

    @property
    def proprioception(self):
        return [
            self.joints_pos, 
            self.appedages_pos,
            self.actuator_activation,
            self.end_effectors_pos,
            self.head_height,
            self.joints_vel,
            self.world_zaxis,
        ] + self._collect_from_attachments('proprioception')