from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from fusion_mppi import mppi
from utils import env_conf, sim_init, data_transfer
import time
import socket
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

# Make the environment and simulation
allow_viewer = True
num_envs = 1 
spacing = 10.0
robot = "point_robot"               # choose from "point_robot", "boxer", "albert"
obstacle_type = "normal"            # choose from "normal", "battery"
control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
gym, sim, viewer, envs, robot_handles = sim_init.make(allow_viewer, num_envs, spacing, robot, obstacle_type, control_type)

# Acquire states
dof_states, num_dofs, num_actors, root_states = sim_init.acquire_states(gym, sim, print_flag=False)

# Time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0

# Set server address
server_address = './uds_socket'

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.connect(server_address)
    while viewer is None or not gym.query_viewer_has_closed(viewer):
        # Send states to mppi
        s.sendall(data_transfer.torch_to_bytes(dof_states))
        s.sendall(data_transfer.torch_to_bytes(root_states))

        # Receive message and optimal action from mppi
        message = s.recv(1024)
        b = s.recv(1024)
        action = data_transfer.bytes_to_torch(b)

        # Visulize the trajectories
        sim_init.visualize_trajs(gym, viewer, envs[0], action, dof_states, frame_count)

        # Apply optimal action
        gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(action))

        # Step the similation
        sim_init.step(gym, sim)
        sim_init.refresh_states(gym, sim)

        # Step rendering
        sim_init.step_rendering(gym, sim, viewer)
        next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

# Destroy the simulation
sim_init.destroy_sim(gym, sim, viewer)