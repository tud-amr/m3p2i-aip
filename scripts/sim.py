from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from fusion_mppi import mppi
from utils import env_conf, sim_init, data_transfer
import time, numpy as np
import socket
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

# Make the environment and simulation
allow_viewer = True
num_envs = 1 
spacing = 10.0
robot = "point_robot"               # choose from "point_robot", "boxer", "albert"
environment_type = "normal"            # choose from "normal", "battery"
control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
dt = 0.05

gym, sim, viewer, envs, robot_handles = sim_init.make(allow_viewer, num_envs, spacing, robot, environment_type, control_type, dt=dt)

# Acquire states
dof_states, num_dofs, num_actors, root_states = sim_init.acquire_states(gym, sim, print_flag=False)

# Time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0
count = 0

# Set server address
server_address = './uds_socket'

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.connect(server_address)
    t_prev = time.monotonic()
    while viewer is None or not gym.query_viewer_has_closed(viewer):
        # Send dof states to mppi and receive message
        s.sendall(data_transfer.torch_to_bytes(dof_states))
        message = s.recv(1024)

        # Send root states and receive optimal actions
        s.sendall(data_transfer.torch_to_bytes(root_states))
        b = s.recv(2**14)
        actions = data_transfer.bytes_to_torch(b)

        # Clear lines at the beginning
        gym.clear_lines(viewer)
        
        # Send message and receive rollout states
        s.sendall(b"Visualize flag")
        visualize_rollouts = s.recv(1024)
        visualize_rollouts = int(data_transfer.bytes_to_torch(visualize_rollouts))
        if visualize_rollouts:
            s.sendall(b"Visualize rollouts")
            K = s.recv(1024)
            K = int(data_transfer.bytes_to_numpy(K))
            rollout_states = np.zeros((1, 2), dtype=np.float32)
            for i in range(K):
                s.sendall(b"next")
                _rollout_state = s.recv(2**18)
                rollout_state = data_transfer.bytes_to_numpy(_rollout_state)
                sim_init.visualize_rollouts(gym, viewer, envs[0], rollout_state)

        # Visualize optimal trajectory
        #sim_init.visualize_traj(gym, viewer, envs[0], actions, dof_states)

        action = actions[0]
        if robot == 'boxer':
            r = 0.08
            L = 2*0.157
            # Diff drive fk
            action_fk = action.clone()
            action_fk[0] = (action[0] / r) - ((L*action[1])/(2*r))
            action_fk[1] = (action[0] / r) + ((L*action[1])/(2*r))
            action = action_fk


        # Apply optimal action
        gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(action))

        # Update movement of dynamic obstacle
        sim_init.update_dyn_obs(gym, sim, num_actors, num_envs, count)
        count += 1

        # Step the similation
        sim_init.step(gym, sim)
        sim_init.refresh_states(gym, sim)

        # Step rendering
        t_now = time.monotonic()
        if (t_now - t_prev) < dt:
            sim_init.step_rendering(gym, sim, viewer, sync_frame_time=True)
        else:
            sim_init.step_rendering(gym, sim, viewer, sync_frame_time=False)
        t_prev = t_now

        next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

# Destroy the simulation
sim_init.destroy_sim(gym, sim, viewer)
