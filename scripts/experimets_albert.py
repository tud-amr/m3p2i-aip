from isaacgym import gymapi
from isaacgym import gymtorch
import torch
from utils import sim_init

# Make the environment and simulation
allow_viewer = True
num_envs = 12
spacing = 10.0
robot = "albert"
control_type = "vel_control"
gym, sim, viewer = sim_init.make(allow_viewer, num_envs, spacing, robot, control_type)

# Acquire states
dof_states, num_dofs, num_actors, root_tensor, saved_root_tensor = sim_init.acquire_states(gym, sim, print_flag=True)

# Time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0

while viewer is None or not gym.query_viewer_has_closed(viewer):
    sim_init.step(gym, sim)

    sim_init.keyboard_control(gym, sim, viewer, robot, num_dofs, num_envs, dof_states)

    sim_init.step_rendering(gym, sim, viewer)
    next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

sim_init.destroy_sim(gym, sim, viewer)