from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from fusion_mppi import mppi
from utils import env_conf, sim_init, data_transfer, skill_utils
import time, numpy as np
import socket
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)
import matplotlib.pyplot as plt

# Make the environment and simulation
log_data = False                    # Set true for plots of control inputs and other stats
allow_viewer = True
num_envs = 1 
spacing = 10.0
control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
dt = 0.05

# Helper variables, same as in fusion_mppi
suction_active = False      # Activate suction or not when close to purple box
block_index = 7
kp_suction = 400

# Time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0
count = 0
sim_time = np.array([])

# Set server address
server_address = './uds_socket'

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.connect(server_address)
    t_prev = time.monotonic()

    s.sendall(b"Robot")
    robot = s.recv(1024).decode()
    s.sendall(b"Environment")
    environment_type = s.recv(1024).decode()

    gym, sim, viewer, envs, robot_handles = sim_init.make(allow_viewer, num_envs, spacing, robot, environment_type, control_type, dt=dt)

    # Acquire states
    dof_states, num_dofs, num_actors, root_states = sim_init.acquire_states(gym, sim, print_flag=False)

    actors_per_env = int(num_actors/num_envs)
    bodies_per_env = gym.get_env_rigid_body_count(envs[0])

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

        if 'action_seq' not in locals():
            action_seq = torch.zeros_like(action)

        action_seq = torch.cat((action_seq, action), 0)
        # Apply optimal action
        gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(action))

        actor_root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        
        if suction_active: 
            root_positions = torch.reshape(actor_root_state[:, 0:2], (num_envs, actors_per_env, 2)) 
            dof_pos = dof_states[:,0].reshape([num_envs, 2])
            # simulation of a magnetic/suction effect to attach to the box
            suction_force, _, _ = skill_utils.calculate_suction(root_positions[:, block_index, :], dof_pos, num_envs, kp_suction, block_index, bodies_per_env)
            print(suction_force)
            # Apply suction/magnetic force
            gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(torch.reshape(suction_force, (num_envs*bodies_per_env, 3))), None, gymapi.ENV_SPACE)

        # Update movement of dynamic obstacle
        if environment_type == 'arena':
            sim_init.update_dyn_obs(gym, sim, num_actors, num_envs, count)
            count += 1

        # Step the similation
        sim_init.step(gym, sim)
        sim_init.refresh_states(gym, sim)

        # Step rendering
        sim_time = np.append(sim_time, t_prev)
        t_now = time.monotonic()
        if (t_now - t_prev) < dt:
            sim_init.step_rendering(gym, sim, viewer, sync_frame_time=True)
        else:
            sim_init.step_rendering(gym, sim, viewer, sync_frame_time=False)
        t_prev = t_now

        next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

# Saving and plotting
if log_data:
    sim_time-= sim_time[0]
    sim_time = np.append(0, sim_time)

    num_dof = int(list(action.size())[0])
    action_seq = action_seq.reshape(len(sim_time), num_dof)
    ctrl_input = np.zeros([len(sim_time), num_dof])

    fig, axs = plt.subplots(num_dof)
    fig.suptitle('Control Inputs')
    plot_colors = ['hotpink','darkviolet','mediumblue']

    if robot == "point_robot" or robot == "heijn":
        label = ['x_vel', 'y_vel', 'theta_vel']
    elif robot == "boxer":
        label = ['r_vel', 'l_vel']
    elif robot == "panda":
        label = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'left_f', 'right_f']
    elif robot == "omni_panda":
        label = ['x_vel', 'y_vel', 'theta_vel','joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'left_f', 'right_f']

    for j in range(num_dof):
        ctrl_input[:,j] = action_seq[:,j].tolist()
        axs[j].plot(sim_time, ctrl_input[:,j], color=plot_colors[1], marker=".")
        axs[j].legend([label[j]])
        axs[j].set(xlabel = 'Time [s]')

    print("Avg. control frequency", len(action_seq)/sim_time[-1])
    plt.show()

# Destroy the simulation
sim_init.destroy_sim(gym, sim, viewer)
