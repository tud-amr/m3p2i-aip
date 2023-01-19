from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
import sys
sys.path.append('../')
from utils import sim_init, skill_utils

# Make the environment and simulation
allow_viewer = True
num_envs = 1
spacing = 10.0
robot = "point_robot"                     # "point_robot", "boxer", "husky", "albert", and heijn
environment_type = "normal"            # choose from "normal", "battery"
control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"

# Helper variables
suction_active = True       # Activate suction or not when close to purple box
block_index = 7
kp_suction = 400

# Time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0
count = 0

gym, sim, viewer, envs, robot_handles = sim_init.make(allow_viewer, num_envs, spacing, robot, environment_type, control_type)

# Acquire states
dof_states, num_dofs, num_actors, _ = sim_init.acquire_states(gym, sim, print_flag=False)
actors_per_env = int(num_actors/num_envs)
bodies_per_env = gym.get_env_rigid_body_count(envs[0])

# Main loop
while viewer is None or not gym.query_viewer_has_closed(viewer):
    # Step the simulation
    sim_init.step(gym, sim)
    sim_init.refresh_states(gym, sim)

    # actor_root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)) # [num_envs*16, 13]
    # root_positions = torch.reshape(actor_root_state[:, 0:2], (num_envs, actors_per_env, 2)) # [num_envs, 16, 2]
    # dof_pos = dof_states[:,0].reshape([num_envs, 2]) # [num_envs, 2]
    
    # if suction_active:
    #     # Simulation of a magnetic/suction effect to attach to the box
    #     suction_force, _, _ = skill_utils.calculate_suction(root_positions[:, block_index, :], dof_pos, num_envs, kp_suction, block_index, bodies_per_env)
    #     # Apply suction/magnetic force
    #     gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(torch.reshape(suction_force, (num_envs*bodies_per_env, 3))), None, gymapi.ENV_SPACE)

    # Respond to keyboard
    sim_init.keyboard_control(gym, sim, viewer, robot, num_dofs, num_envs, dof_states, control_type)

    # Step rendering
    sim_init.step_rendering(gym, sim, viewer)
    next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

# Destroy the simulation
sim_init.destroy_sim(gym, sim, viewer)
