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
control_type = "vel_control"
gym, sim, viewer = sim_init.make(allow_viewer, num_envs, spacing, robot, control_type)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
num_dofs = gym.get_sim_dof_count(sim)

# time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0

# MPPI settings
step = 0
mppi_step_count = 100

_net_cf = gym.acquire_net_contact_force_tensor(sim)
net_cf = gymtorch.wrap_tensor(_net_cf)

while viewer is None or not gym.query_viewer_has_closed(viewer):
    sim_init.step(gym, sim)
    step += 1

    _net_cf = gym.refresh_net_contact_force_tensor(sim)

    sim_init.keyboard_control(gym, sim, viewer, robot, num_dofs, num_envs, dof_states)
    
    # Net contact forces with black walls. Indexes are according to how you add the actors in the env
    if torch.max(net_cf[0:4])>1 or torch.max(net_cf[0:4])<-1:
        print("Collision")
    else:
        print("Ok")

    sim_init.step_rendering(gym, sim, viewer)
    next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

sim_init.destroy_sim(gym, sim, viewer)