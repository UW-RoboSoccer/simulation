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

Verify setup by running the visualize.py file
```
python visualize.py
```

The result should generate 500 frames of a simulation of a humanoid and then open a viewer that renders these frames on loop. 

Close the window to end the program.

## Development