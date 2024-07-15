<p align="center">
    <h1 align="center">Multi-Modal MPPI and Active Inference for Reactive Task and Motion Planning</h1>
    <h3 align="center"><a href="https://arxiv.org/abs/2312.02328">Paper</a> | <a href="https://autonomousrobots.nl/paper_websites/m3p2i-aip">Website</a> | <a href="https://www.youtube.com/watch?v=y2CTgv6hxVI&t=2s">Video</a> </h3>
</p>

Repository for reactive task and motion planning making use of active inference for symbolic planning, and a new multi-modal MPPI for motion planning. Rollouts are evaluated in IsaacGym, a parallelizable physics simulator.

<p align="center">
    <img src="./src/m3p2i_aip/assets/images/m3p2i_1corner.gif" alt="001" width=370 /> &nbsp; <img src="./src/m3p2i_aip/assets/images/m3p2i_pick_with_obs.gif" alt="002" width=370 />
</p>


## Installation
First, clone the repo in your folder and create the conda environment. 
````bash
cd <project_folder>
git clone https://github.com/tud-amr/m3p2i-aip.git

conda create -n m3p2i-aip python=3.8
conda activate m3p2i-aip
````

This project requires the source code of IsaacGym. Download it from https://developer.nvidia.com/isaac-gym, unzip and paste it in the `thirdparty` folder. Move to IsaacGym and install the package.
````bash
cd <project_folder>/m3p2i-aip/thirdparty/IsaacGym_Preview_4_Package/isaacgym/python
pip install -e. 
````

Then install the current package by:
````bash
cd <project_folder>/m3p2i-aip
pip install -e. 
````

Now you are ready to test an example file, where you can drive the robot around with ASDW keys.

````bash
cd <project_folder>/m3p2i-aip/examples
python3 example_key.py
````

## Run the scripts

If you want to test the MPPI, you will need two instances of Isaac Gym, one for the rollouts, and one for the "real system". Run the commands below in two terminals from the `scripts` folder: 
````
python3 reactive_tamp.py --robot $(robot_type) --task $(task_type)
````

````
python3 sim.py --robot $(robot_type) --task $(task_type)
````

You can specify `robot_type` as `point`, `boxer`, `heijn` or `panda`, and specify `task_type` as `simple`, `patrolling`, `reactive`, `pick` or `reactive_pick`. You can also try experiments with the arguments passed to the MPPI, such as sampling around prior controllers or null actions, as well as time horizon and number of samples, which can be modified in the */params* folder. 

## Cite

If you find the code useful, please cite:
```
@article{zhang2024multi,
  title={Multi-Modal MPPI and Active Inference for Reactive Task and Motion Planning},
  author={Zhang, Yuezhe and Pezzato, Corrado and Trevisan, Elia and Salmi, Chadi and Corbato, Carlos Hern{\'a}ndez and Alonso-Mora, Javier},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```

## Related works

* **Isaac-mppi**: an MPPI implementation that uses IsaacGym as a dynamic model ([paper](https://arxiv.org/abs/2307.09105), [website](https://sites.google.com/view/mppi-isaac/), [code](https://github.com/tud-airlab/mppi-isaac)).
* **Biased-mppi**: an MPPI implementation whose sampling distribution is informed with ancillary controllers ([paper](https://ieeexplore.ieee.org/document/10520879), [website](https://autonomousrobots.nl/paper_websites/biased-mppi), [code](https://github.com/eliatrevisan/biased-mppi)).
* **AIP**: an Active Inference planner for decision making ([paper](https://ieeexplore.ieee.org/document/10004745), [video](https://www.youtube.com/watch?v=dEjXu-sD1SI), [code](https://github.com/cpezzato/decision_making)).