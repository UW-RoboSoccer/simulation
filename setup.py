from setuptools import setup, find_packages

setup(
    name="soccer_env",
    verions="0.0.1",
    description="A soccer environment for reinforcement learning",
    long_description=open("README.md").read(), 
    author="University of Waterloo RoboSoccer",
    autho_email="uwrobosoccer@gmail.com",
    url="https://uw-robosoccer.github.io/",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "mujoco",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)