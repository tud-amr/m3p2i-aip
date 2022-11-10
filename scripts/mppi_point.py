from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from fusion_mppi import mppi, fusion_mppi
from utils import env_conf, sim_init, data_transfer
import time
import copy
import socket
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

# Make the environment and simulation
allow_viewer = False
num_envs = 2000
spacing = 10.0
robot = "point_robot"               # choose from "point_robot", "boxer", "albert"
obstacle_type = "normal"            # choose from "normal", "battery"
control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
gym, sim, viewer, envs, robot_handles = sim_init.make(allow_viewer, num_envs, spacing, robot, obstacle_type, control_type)

# Acquire states
dof_states, num_dofs, num_actors, root_states = sim_init.acquire_states(gym, sim, print_flag=False)

# Creater mppi object
mppi = fusion_mppi.FUSION_MPPI(
    dynamics=None, 
    running_cost=None, 
    nx=2, 
    noise_sigma = torch.tensor([[2, 0], [0, 2]], device="cuda:0", dtype=torch.float32),
    num_samples=num_envs, 
    horizon=10,
    lambda_=0.1, 
    device="cuda:0", 
    u_max=torch.tensor([3.0, 3.0]),
    u_min=torch.tensor([-3.0, -3.0]),
    step_dependent_dynamics=True,
    terminal_state_cost=None
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
            while True:
                res = conn.recv(1024)
                if not res == b'': break
            r = copy.copy(res)
            _dof_states = data_transfer.bytes_to_torch(r).repeat(num_envs, 1)

            # Send message
            conn.sendall(b"next please")

            # Receive root states
            while True:
                res = conn.recv(2**14)
                if not res == b'': break
            r = copy.copy(res)
            _root_states = data_transfer.bytes_to_torch(r).repeat(num_envs, 1)

            # Reset the simulator to requested state
            s = _dof_states.view(-1, 4) # [x, v_x, y, v_y]
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(s))
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(_root_states))
            sim_init.refresh_states(gym, sim)

            # Update gym in mppi
            mppi.update_gym(gym, sim)

            # Compute optimal action and send to real simulator
            action = mppi.command(s)
            conn.sendall(data_transfer.torch_to_bytes(action))