# Thirdparty software

Isaacgym is not accessible through pypi, but through https://developer.nvidia.com/isaac-gym. Download and unzip the software and paste it into this folder.


## Install IsaacGym

**Prerequisites:**
* Ubuntu 18.04 or 20.04.
* Python 3.6, 3.7 or 3.8.
* Minimum NVIDIA driver version:
    * Linux: 470.

````bash
cd IsaacGym_Preview_4_Package/isaacgym/python
pip install -e. 
````
## Install IsaacGymEnvs (optional)
In order to use the RL examples, please clone the IsaacGymEnvs repo into this folder by:
````bash
git clone https://github.com/isaac-sim/IsaacGymEnvs.git
````
After Isaacgym is working in the current python environment, then install IsaacGymEnvs package by:
````bash
cd IsaacGymEnvs
pip install -e .
````

## Troubleshooting
If you have an Nvidia card and after running the simulation you get a black screen, you might need to force the use of the GPU card through ``export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json``. Run this command from the same folder as the script to be launched

If you encounter ``AttributeError: module 'numpy' has no attribute 'float'``, just change `np.float` to `np.float32` in the code.