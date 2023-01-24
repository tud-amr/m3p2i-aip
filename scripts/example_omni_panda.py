from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from utils import sim_init, skill_utils

# Make the environment and simulation
allow_viewer = True
num_envs = 100
spacing = 1.5
robot = "omni_panda"                     # "point_robot", "boxer", "husky", "albert", and "heijn", "panda"
environment_type = "table"            # choose from "normal", "battery"
control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"

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

    # Step rendering
    sim_init.step_rendering(gym, sim, viewer)
    next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

# Destroy the simulation
sim_init.destroy_sim(gym, sim, viewer)
