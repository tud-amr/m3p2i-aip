from isaacgym import gymapi
from isaacgym import gymtorch
import torch
from utils import sim_init
import numpy as np 

# Make the environment and simulation
allow_viewer = True
num_envs = 128
spacing = 10.0
robot = "boxer"
obstacle_type = "normal"
control_type = "vel_control"
gym, sim, viewer, _, _ = sim_init.make(allow_viewer, num_envs, spacing, robot, obstacle_type, control_type)

# Acquire states
dof_states, num_dofs, num_actors, root_tensor, saved_root_tensor = sim_init.acquire_states(gym, sim, print_flag=True)

# Time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0

_net_cf = gym.acquire_net_contact_force_tensor(sim)
net_cf = gymtorch.wrap_tensor(_net_cf)

while viewer is None or not gym.query_viewer_has_closed(viewer):
    # Step the simulation
    sim_init.step(gym, sim)
    sim_init.refresh_states(gym, sim)

    # Respond to keyboard
    sim_init.keyboard_control(gym, sim, viewer, robot, num_dofs, num_envs, dof_states)
    
    # Net contact forces with black walls. Indexes are according to how you add the actors in the env
    _net_cf = gym.refresh_net_contact_force_tensor(sim)
    if torch.max(net_cf[0:4])>1 or torch.max(net_cf[0:4])<-1:
        print("Collision")
    else:
        pass
    
    # Step rendering
    sim_init.step_rendering(gym, sim, viewer)
    next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

# Destroy the simulation
sim_init.destroy_sim(gym, sim, viewer)