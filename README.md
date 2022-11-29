# Reactive TAMP and controller fusion

Repositiry for reactive task and motion planning making use of active inference and behavior trees for symbolic planning, and MPPI for motion planning. Rollouts are evaluated in Isaac Gym, a parallelizable physics simulator.

There are also a number of examples to get familiar with Isaac Gym with our robots. 

## Status
Under development

### Requirements
- Python 3.6, <3.10
- IsaacGym 
- Nvidia graphics card

This package has been tested in Ubuntu 2020.

### Dependencies
Some examples in the folder */scripts* use the classes and methods of the package [decision_making](https://gitlab.tudelft.nl/airlab-delft/ng-staging/controller_fusion/decision_making). Please refer to [decision_making](https://gitlab.tudelft.nl/airlab-delft/ng-staging/controller_fusion/decision_making) for installation instruction if you wish to use those.

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

If you want to test the MPPI, you will need two instances of Isaac Gym, one for the rollouts, and one for the "real system". Run the commands below in two terminals from the */scripts* folder: 
````
python3 mppi_$(robot_type).py 
````

````
python3 sim.py
````

## Troubleshooting
If you have an Nvidia card and after running the simulation you get a black screen, you might need to force the use of the GPU card through ``export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json``. Run this command from the same folder as the script to be launched