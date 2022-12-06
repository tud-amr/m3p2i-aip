from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from utils import sim_init

# Make the environment and simulation
allow_viewer = True
num_envs = 4
spacing = 10.0
robot = "point_robot"                     # "point_robot", "boxer", "husky", and "albert"
environment_type = "normal"            # choose from "normal", "battery"
control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
gym, sim, viewer, _, _ = sim_init.make(allow_viewer, num_envs, spacing, robot, environment_type, control_type)

# Acquire states
dof_states, num_dofs, num_actors, root_states = sim_init.acquire_states(gym, sim, print_flag=True)

# Time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0
count = 0

# Main loop
while viewer is None or not gym.query_viewer_has_closed(viewer):
    # Step the simulation
    sim_init.step(gym, sim)
    sim_init.refresh_states(gym, sim)

    # Respond to keyboard
    sim_init.keyboard_control(gym, sim, viewer, robot, num_dofs, num_envs, dof_states, control_type)

    # Update dynamic obstacle
    sim_init.update_dyn_obs(gym, sim, num_actors, num_envs, count)
    count += 1

    # Step rendering
    sim_init.step_rendering(gym, sim, viewer)
    next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

# Destroy the simulation
sim_init.destroy_sim(gym, sim, viewer)