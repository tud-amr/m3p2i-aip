# Repository for decision making through active infrence and behavior trees

This is a collection of examples for later use of MPPI, isaac gym, and active inference

## Status
Under development

### Requirements
- Python 3.6, <3.10
- IsaacGym 
- Nvidia graphics card

This package has been tested in Ubuntu 2020.

### Dependencies
Some examples in the folder */examples* use the classes and methods of the package [decision_making](https://gitlab.tudelft.nl/airlab-delft/ng-staging/controller_fusion/decision_making). Please refer to [decision_making](https://gitlab.tudelft.nl/airlab-delft/ng-staging/controller_fusion/decision_making) for installation instruction if you wish to use those.

## How to use this repo
Just clone the repo in a location in you computer. 

*Skip this passage if you installed IsaacGym at system level*. If you have a Conda environment for IsaacGym, activate the environment

````bash
cd <you_isaac_gym_folder>
conda acivate <env_name>
````

Then you are ready to test a script:

````bash
cd <path/to/your/isaac_gym_folder>/scripts
python3 example_key.py
````
With this script you can drive the robot around with ASDW keys. 

## Troubleshooting
If you hav an Nvidia card and after running the simulation you get a black screen, you might need to force the use of the GPU card through ``export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json``. Run this command from the same folder as the script to be launched