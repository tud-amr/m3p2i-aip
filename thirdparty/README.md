# Thirdparty software

Isaacgym is not accessible through pypi, but through https://developer.nvidia.com/isaac-gym. Download and unzip the software and paste it into this folder.

In order to use the RL examples, please clone the IsaacGymEnvs repo into this folder by:
```
git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
```
After Isaacgym is working in the current python environment, then install IsaacGymEnvs package by:
```
cd IsaacGymEnvs
pip install -e .
```
TODO: this line can be integrated in the poetry installation.