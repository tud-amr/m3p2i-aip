# Multi-Modal MPPI and Active Inference for Reactive Task and Motion Planning

Repositiry for reactive task and motion planning making use of active inference and behavior trees for symbolic planning, and MPPI for motion planning. Rollouts are evaluated in Isaac Gym, a parallelizable physics simulator.

There are also a number of examples to get familiar with Isaac Gym with our robots. 

## Status
Under development

## Project website 
https://sites.google.com/view/m3p2i-aip 

### Requirements
- Python 3.6, <3.10
- IsaacGym 
- Nvidia graphics card

This package has been tested in Ubuntu 2020.

### Dependencies
Some examples in the folder */scripts* use the classes and methods of the package [decision_making](https://gitlab.tudelft.nl/airlab-delft/ng-staging/controller_fusion/decision_making). Please refer to [decision_making](https://gitlab.tudelft.nl/airlab-delft/ng-staging/controller_fusion/decision_making) for installation instruction if you wish to use those.

Some other examples use [Optimization Fabrics](https://github.com/maxspahn/fabrics), currently working version is "0.4.1". Please follow installation intructions in the linked repository. 

## How to use this repo
Just clone the repo in a location in you computer. 

*Skip this passage if you installed IsaacGym at system level*. If you have a Conda environment for IsaacGym, activate the environment

````bash
cd <you_isaac_gym_folder>
conda acivate <env_name>
````

Then you are ready to test a script:

````bash
cd <path/to/your/isaac_gym_folder>/scripts/examples
python3 example_key.py
````
With this script you can drive the robot around with ASDW keys. 

If you want to test the MPPI, you will need two instances of Isaac Gym, one for the rollouts, and one for the "real system". Run the commands below in two terminals from the */scripts* folder: 
````
python3 reactive_tamp.py --robot $(robot_type) --task $(task_type)
````

````
python3 sim.py --robot $(robot_type) --task $(task_type)
````

You can specify `robot_type` as `point`, `boxer`, `heijn` or `panda`, and specify `task_type` as `simple`, `patrolling`, `reactive`, `pick` or `reactive_pick`. You can also try experiments with the arguments passed to the MPPI, such as sampling around prior controllers or null actions, as well as time horizon and number of samples, which can be modified in the */params* folder. 

## Troubleshooting
If you have an Nvidia card and after running the simulation you get a black screen, you might need to force the use of the GPU card through ``export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json``. Run this command from the same folder as the script to be launched
