# soccer_env/__init__.py

import os
from .core import *

# Set a base directory for assets
# ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Import submodules
# from soccer_env.core.humanoid import Player, PlayerObservables
# from soccer_env.core.pitch import Pitch
# from soccer_env.core.soccer_ball import SoccerBall
# from soccer_env.env import SoccerEnvironment

__all__ = [ "Pitch", "SoccerBall"]
