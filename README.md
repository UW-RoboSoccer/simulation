# Brax Sim Environment with MJX
Official UWRS Repo for MuJoCo Soccer Simulation

## System Requirements
- WSL Ubuntu 22.04
- CUDA
- NVIDIA CUDA capable GPU
- Python 3.12

## Setup
Make python virtual environment
```
python3 -m venv .venv
```

Install requirements.txt
```
pip install -r requirements.txt
```

Upgrade JAX package to use CUDA
```
pip install --upgrade jax[cuda<version>]
```
**Replace <version> with your CUDA version**

## Visualize
Verify setup by running the visualize.py file
```
python visualize.py
```

This will generate a simulation rollout and then display an opencv window with the simulation. A video named `output.mp4` will be saved in the root directory.

## Troubleshooting
### WSL Ubuntu 22.04
If you are using WSL Ubuntu 22.04, you may encounter issues with rendering the simulation:

**1) opencv: Could not load the Qt platform plugin "xcb" in "" even though it was found**
- To fix this issue be sure to remove any PyQt installations in the python virtual environment
- If the issue persists install the headless version of opencv along with opencv-python
```
pip install opencv-python-headless
```

**2) Visual artifcats in video such as black bars or flickering**
- If you have a NVIDIA GPU WSL may have issues with rendering the simulation. To fix this issue you can export the software openGL environment variable
```
export LIBGL_ALWAYS_SOFTWARE=1
```

**3) Verifying Display connection from WSL to Host**
- If you are using VcXsrv to connect WSL/VM to your host Windows system try the following in the WSL terminal
- Use XLaunch and configure the settings and check the box for Disable Access Control. Then check the Display setting
```
export DISPLAY=:0
```
- Use this to verify that the IP address of your display is correct. If 'nameserver:' appears before the IP address - something is incorrectly configured
```
echo $DISPLAY
```
- When you're done you should be able to run this default GUI. It should show a window of 3 spinning gears
```
glxgears
```
