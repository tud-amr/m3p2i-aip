from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from utils import sim_init

# Make the environment and simulation
allow_viewer = True
num_envs = 4
spacing = 10.0
robot = "point_robot"
control_type = "vel_control"
gym, sim, viewer = sim_init.make(allow_viewer, num_envs, spacing, robot, control_type)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
num_dofs = gym.get_sim_dof_count(sim)
num_actors = gym.get_sim_actor_count(sim)

# acquire root state tensor descriptor and wrap it in a PyTorch Tensor
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
root_tensor = gymtorch.wrap_tensor(_root_tensor)
saved_root_tensor = root_tensor.clone()

# get relevant info
print("root_tensor", root_tensor.size())
print('number of DOFs:', num_dofs) # num_envs * dof_per_actor
print("dof_state size:", dof_states.size()) # [num_dofs, 2]
print("pos", dof_states[:,0])
print("vel", dof_states[:,1])
print("actor num", num_actors)

# time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0
step = 0

while viewer is None or not gym.query_viewer_has_closed(viewer):
    sim_init.step(gym, sim)
    step += 1

    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    sim_init.keyboard_control(gym, sim, viewer, robot, num_dofs, num_envs, dof_states, control_type)

    sim_init.step_rendering(gym, sim, viewer)
    next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

sim_init.destroy_sim(gym, sim, viewer)