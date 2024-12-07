from dm_control.composer import arena
from dm_control import mjcf
import os
from dm_control.viewer import launch

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
_XML_PATH = os.path.join(parent_dir, 'assets', 'flat_plane.xml')

class FlatPlane(arena.Arena):
    """A flat plane arena."""

    def _build(self, _xml_path=_XML_PATH, name='flat_plane'):
        """Builds this arena.

        Args:
          name: The name of this entity.
        """
        # self.mjcf_model.worldbody.add('geom', type='plane', size=(1, 1, 0.1), rgba=(0.8, 0.6, 0.4, 1))
        # super()._build(name=name)
        self._xml_path = _xml_path

        self._mjcf_root = mjcf.from_path(self._xml_path)
        if name:
            self._mjcf_root.model = name


# if __name__ == '__main__':
#     flatplane = FlatPlane()
#     print(FlatPlane.mjcf_model)
    