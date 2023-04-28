from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
import sys
sys.path.append('../')
from utils import sim_init, skill_utils

class Params:
    def __init__(self) -> None:
        # Make the environment and simulation
        self.allow_viewer = True
        self.num_envs = 1
        self.spacing = 10.0
        self.robot = "panda"          # "point_robot", "boxer", "husky", "albert", "panda", and "heijn"
        self.environment_type = "cube"    # "normal", "battery"
        self.control_type = "vel_control"   # "vel_control", "pos_control", "force_control"


        # Helper variables
        self.suction_active = False       # Activate suction or not when close to purple box
        self.block_index = "None"
        self.kp_suction = 400

        # Time logging
        self.frame_count = 0
        self.next_fps_report = 2.0
        self.t1 = 0
        self.count = 0

        # Print info
        self.print_flag = False

# Get parameters
params = Params()

# Make the environment
gym, sim, viewer, envs, robot_handles = sim_init.make(params.allow_viewer, 
                                                      params.num_envs, 
                                                      params.spacing, 
                                                      params.robot, 
                                                      params.environment_type, 
                                                      params.control_type)

# Acquire states
states_dict = sim_init.acquire_states(gym, sim, params)
dof_states = states_dict["dof_states"]
num_dofs = states_dict["num_dofs"]
dofs_per_robot = states_dict["dofs_per_robot"]
actors_per_env = states_dict["actors_per_env"]
bodies_per_env = states_dict["bodies_per_env"]
block_pos = states_dict["block_pos"]
robot_pos = states_dict["robot_pos"]

# Main loop
while viewer is None or not gym.query_viewer_has_closed(viewer):
    # Step the simulation
    sim_init.step(gym, sim)
    sim_init.refresh_states(gym, sim)

    if params.suction_active:
        # Simulation of a magnetic/suction effect to attach to the box
        suction_force, _, _ = skill_utils.calculate_suction(block_pos, robot_pos, params.num_envs, params.kp_suction, params.block_index, bodies_per_env)
        # print('suc', suction_force)
        # Apply suction/magnetic force
        gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(torch.reshape(suction_force, (params.num_envs*bodies_per_env, 3))), None, gymapi.ENV_SPACE)

    # Respond to keyboard
    sim_init.keyboard_control(gym, sim, viewer, params.robot, num_dofs, params.num_envs, dof_states, params.control_type)

    # Step rendering
    sim_init.step_rendering(gym, sim, viewer)
    params.next_fps_report, params.frame_count, params.t1 = sim_init.time_logging(gym, sim, params.next_fps_report, params.frame_count, params.t1, params.num_envs)

# Destroy the simulation
sim_init.destroy_sim(gym, sim, viewer)