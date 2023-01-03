from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from utils import sim_init

# Make the environment and simulation
allow_viewer = True
num_envs = 1
spacing = 10.0
robot = "point_robot"                     # "point_robot", "boxer", "husky", and "albert"
environment_type = "normal"            # choose from "normal", "battery"
control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"

gym, sim, viewer, envs, robot_handles = sim_init.make(allow_viewer, num_envs, spacing, robot, environment_type, control_type)

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 6, 8), gymapi.Vec3(1.5, 0, 0))
gym.prepare_sim(sim)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = torch.reshape(gymtorch.wrap_tensor(_dof_states), (num_envs, 4))

num_dofs = gym.get_sim_dof_count(sim)

# Time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0
count = 0

bodies_per_env = 18
forces = torch.zeros((num_envs, bodies_per_env, 3), dtype=torch.float32, device='cuda:0', requires_grad=False)
box_index = 7
robot_index = 15
max_suction = 100

def calculate_suction(v1, v2):
    dir_vector = v1 - v2
    magnitude = 1/torch.tensor([torch.linalg.norm(dir_vector**2)], device='cuda:0', dtype=torch.float)
    direction = dir_vector/magnitude
    # if torch.isnan(magnitude):
    #     print("ERROR: suction is NAN")
    # elif magnitude**2 < 1e-9:
    #     print("ERROR: suction too close")
    if magnitude > 5:
        magnitude = 0
    print('magnitude', magnitude)
    print('direction', direction)
    force = (magnitude**2)*direction
    return force

kp = 200
# Main loop
while viewer is None or not gym.query_viewer_has_closed(viewer):
    # Step the simulation
    sim_init.step(gym, sim)
    sim_init.refresh_states(gym, sim)

    actor_root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    root_positions = torch.reshape(actor_root_state[:, 0:2], (num_envs, bodies_per_env-2, 2))
    dof_pos = dof_states[:,[0,2]]

    suction = calculate_suction(root_positions[:, box_index, :], dof_pos)

    forces[:,box_index, 0] = -kp*suction[:,0]
    forces[:,box_index, 1] = -kp*suction[:,1]

    # Limit forces
    forces = torch.clamp(forces, min=-500, max=500)
    print(forces[:,box_index, 0:2])
    # apply actions
    gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(torch.reshape(forces, (num_envs*bodies_per_env, 3))), None, gymapi.LOCAL_SPACE)
    
    # Respond to keyboard
    sim_init.keyboard_control(gym, sim, viewer, robot, num_dofs, num_envs, dof_states, control_type)

    # Step rendering
    sim_init.step_rendering(gym, sim, viewer)
    next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

# Destroy the simulation
sim_init.destroy_sim(gym, sim, viewer)
