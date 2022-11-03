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

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
num_dofs = gym.get_sim_dof_count(sim)
print('Number of DOFs:', num_dofs) # num_envs * 13

# time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0
step = 0

while viewer is None or not gym.query_viewer_has_closed(viewer):
    sim_init.step(gym, sim)
    step += 1

    sim_init.keyboard_control(gym, sim, viewer, robot, num_dofs, num_envs, dof_states)

    sim_init.step_rendering(gym, sim, viewer)
    next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

sim_init.destroy_sim(gym, sim, viewer)