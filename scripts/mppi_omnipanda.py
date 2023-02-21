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
device = "cuda:0"
num_envs = 300  # 50 is borderline acceptable behavior
spacing = 2.0
robot = "omni_panda"                     # choose from "point_robot", "boxer", "albert", "panda"
environment_type = "store"          # choose from "arena", "battery", "store"
control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
dt = 0.01
substeps = 1

gym, sim, viewer, envs, robot_handles = sim_init.make(allow_viewer, num_envs, spacing, robot, environment_type, control_type, dt=dt, substeps=substeps)
# Acquire states
dof_states, num_dofs, num_actors, root_states = sim_init.acquire_states(gym, sim, print_flag=False)
actors_per_env = int(num_actors/num_envs)
bodies_per_env = gym.get_env_rigid_body_count(envs[0])

# For storm mppi mode
sigma = 2
sigma_base = 5
max_vel = 2
max_vel_base = 8
max_vel_finger = 0.3
sigma_finger = 0.5
inv_temp = 0.1

# For pure-randm mppi
# sigma = 1
# sigma_base = 7
# max_vel = 1
# max_vel_base = 2
# inv_temp = 0.05

# Creater mppi object
mppi = fusion_mppi.FUSION_MPPI(
    dynamics=None, 
    running_cost=None, 
    nx=24, 
    noise_sigma = torch.tensor([[sigma_base, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, sigma_base, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, sigma_base, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, sigma, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0,0, sigma, 0, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0,0, 0, sigma, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0,0, 0, 0, sigma, 0, 0, 0, 0, 0],
                                [0, 0, 0,0, 0, 0, 0, sigma, 0, 0, 0, 0],
                                [0, 0, 0,0, 0, 0, 0, 0, sigma, 0, 0, 0],
                                [0, 0, 0,0, 0, 0, 0, 0, 0, sigma, 0, 0],
                                [0, 0, 0,0, 0, 0, 0, 0, 0, 0, sigma_finger, 0],
                                [0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, sigma_finger],], device=device, dtype=torch.float32),
    num_samples=num_envs, 
    horizon=20,
    lambda_=inv_temp, 
    device=device, 
    u_max=torch.tensor([max_vel_base, max_vel_base, max_vel_base, max_vel, max_vel, max_vel, max_vel, max_vel, max_vel, max_vel, max_vel_finger, max_vel_finger]),
    u_min=torch.tensor([-max_vel_base, -max_vel_base, -max_vel_base, -max_vel, -max_vel, -max_vel, -max_vel, -max_vel, -max_vel, -max_vel, -max_vel_finger, -max_vel_finger]),
    step_dependent_dynamics=True,
    terminal_state_cost=None,
    sample_null_action=True,
    use_priors=False,
    use_vacuum = False,
    robot_type=robot,
    u_per_command=10,
    actors_per_env=actors_per_env,
    env_type=environment_type,
    bodies_per_env=bodies_per_env,
    filter_u=True
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
        
        # Send info for simulation, robot and environment types
        res = conn.recv(1024)
        conn.sendall(robot.encode())

        res = conn.recv(1024)
        conn.sendall(environment_type.encode())

        res = conn.recv(1024)
        conn.sendall(data_transfer.numpy_to_bytes(dt))

        res = conn.recv(1024)
        conn.sendall(data_transfer.numpy_to_bytes(substeps))

        i=0
        while True:
            i+=1
            # Receive dof states
            res = conn.recv(2**14)
            r = copy.copy(res)
            _dof_states = data_transfer.bytes_to_torch(r).repeat(num_envs, 1)
            conn.sendall(b"next please")

            # Receive root states
            res = conn.recv(2**14)
            r = copy.copy(res)
            _root_states = data_transfer.bytes_to_torch(r).repeat(num_envs, 1)

            # Reset the simulator to requested state
            s = _dof_states.view(-1, 24) 
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(s))
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(_root_states))
            sim_init.refresh_states(gym, sim)

            # Update gym in mppi
            mppi.update_gym(gym, sim, viewer)

            # Compute optimal action and send to real simulator
            actions = mppi.command(s[0])
            conn.sendall(data_transfer.torch_to_bytes(actions))

            # Send rollouts data
            res = conn.recv(1024)
            conn.sendall(data_transfer.torch_to_bytes(int(visualize_rollouts)))
            if visualize_rollouts:
                # Get the rollouts trajectory
                rollouts = mppi.ee_states[:, :, :].cpu().clone().numpy()
                current_traj = np.zeros((mppi.T, 3))
                K = mppi.K
                res = conn.recv(1024)
                conn.sendall(data_transfer.numpy_to_bytes(mppi.K))
                for i in range(K):
                    res4 = conn.recv(1024)
                    current_traj[:, 0] = rollouts[i][:, 0]     # x pos
                    current_traj[:, 1] = rollouts[i][:, 1]     # y pos
                    current_traj[:, 2] = rollouts[i][:, 2]     # z pos
                    conn.sendall(data_transfer.numpy_to_bytes(current_traj))