from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from fusion_mppi import mppi, fusion_mppi
from utils import env_conf, sim_init, data_transfer
import time
import copy
import socket, io
import numpy as np
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

# Make the environment and simulation
allow_viewer = False
visualize_rollouts = False
num_envs = 200
spacing = 10.0
robot = "boxer"               # choose from "point_robot", "boxer", "albert"
environment_type = "normal"         # choose from "normal", "battery"
control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
gym, sim, viewer, envs, robot_handles = sim_init.make(allow_viewer, num_envs, spacing, robot, environment_type, control_type, dt=0.05)

# Acquire states
dof_states, num_dofs, num_actors, root_states = sim_init.acquire_states(gym, sim, print_flag=False)
actors_per_env = int(num_actors/num_envs)
bodies_per_env = gym.get_env_rigid_body_count(envs[0])

# Creater mppi object
mppi = fusion_mppi.FUSION_MPPI(
    dynamics=None, 
    running_cost=None, 
    nx=4, 
    noise_sigma = torch.tensor([[10, 0], [0, 10]], device="cuda:0", dtype=torch.float32),
    num_samples=num_envs, 
    horizon=15,
    lambda_=0.1, 
    device="cuda:0", 
    u_max=torch.tensor([1.5, 5.5]),
    u_min=torch.tensor([-1.5, -5.5]),
    step_dependent_dynamics=True,
    terminal_state_cost=None,
    sample_null_action=True,
    use_priors=False,
    use_vacuum=False,
    robot_type=robot,
    u_per_command=15,
    actors_per_env=actors_per_env,
    env_type=environment_type,
    bodies_per_env=bodies_per_env
    )

# Make sure the socket does not already exist
server_address = './uds_socket'
data_transfer.check_server(server_address)

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    # Build the connection
    s.bind(server_address)
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        i=0
        while True:
            i+=1
            # Receive dof states
            res = conn.recv(2**16)
            r = copy.copy(res)
            _dof_states = data_transfer.bytes_to_torch(r).repeat(num_envs, 1)
            conn.sendall(b"next please")

            # Receive root states
            res = conn.recv(2**14)
            r = copy.copy(res)
            _root_states = data_transfer.bytes_to_torch(r).repeat(num_envs, 1)

            # Reset the simulator to requested state
            s = _dof_states.view(-1, 4) # [x, v_x, y, v_y]
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(s))
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(_root_states))
            sim_init.refresh_states(gym, sim)

            # Update gym in mppi
            mppi.update_gym(gym, sim, viewer)

            # Compute optimal action and send to real simulator
            actions = mppi.command(s)
            conn.sendall(data_transfer.torch_to_bytes(actions))
            # Send rollouts data
            res = conn.recv(1024)
            conn.sendall(data_transfer.torch_to_bytes(int(visualize_rollouts)))
            # from point robto
            # if visualize_rollouts:
            #     # Get the rollouts trajectory
            #     rollouts = mppi.states[0, :, :, :].cpu().clone().numpy()
            #     current_traj = np.zeros((mppi.T, 2))
            #     K = mppi.K
            #     res = conn.recv(1024)
            #     conn.sendall(data_transfer.numpy_to_bytes(mppi.K))
            #     for i in range(K):
            #         res4 = conn.recv(1024)
            #         current_traj[:, 1] = rollouts[i][:, 0]     # x pos
            #         current_traj[:, 0] = rollouts[i][:, 2]     # y pos
            #         conn.sendall(data_transfer.numpy_to_bytes(current_traj))

