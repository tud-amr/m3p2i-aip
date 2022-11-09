from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from pytorch_mppi import mppi
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

def mppi_dynamics(input_state, action, t):
    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(action))
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_dof_state_tensor(sim)

    if allow_viewer:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

    res = torch.clone(dof_states).view(-1, 4)
    return res

def running_cost(state, action):
    # State: for each environment, the current state containing position and velocity
    # Action: same but for control input
    
    state_pos = torch.cat((state[:, 0].unsqueeze(1), state[:, 2].unsqueeze(1)), 1)
    task_cost = torch.linalg.norm(state_pos - torch.tensor([3, -3], device="cuda:0"), axis=1)
    
    control_cost = torch.sum(torch.square(action),1)
    w_u = 0.01
    # Contact forces
    _net_cf = gym.acquire_net_contact_force_tensor(sim)
    net_cf = gymtorch.wrap_tensor(_net_cf)
    _net_cf = gym.refresh_net_contact_force_tensor(sim)
    # Take only forces in x,y in modulus for each environment. Avoid all collisions
    net_cf = torch.sum(torch.abs(torch.cat((net_cf[:, 0].unsqueeze(1), net_cf[:, 1].unsqueeze(1)), 1)),1)
    coll_cost = torch.sum(net_cf.reshape([num_envs, int(net_cf.size(dim=0)/num_envs)]), 1)
    w_c = 10000 # Weight for collisions
    # Binary check for collisions. So far checking all collision of all actors. TODO: check collision of robot body only       
    coll_cost[coll_cost>0.1] = 1
    coll_cost[coll_cost<=0.1] = 0
    
    return  task_cost + w_u*control_cost + w_c*coll_cost

def terminal_state_cost(states, actions):
    # States: for each environment and for the whole time horizon, the state trajectory containing position and velocity
    # Actions: same but for control input
    return torch.zeros(num_envs, device="cuda:0")

mppi = mppi.MPPI(
    dynamics=mppi_dynamics, 
    running_cost=running_cost, 
    nx=2, 
    noise_sigma = torch.tensor([[2, 0], [0, 2]], device="cuda:0", dtype=torch.float32),
    num_samples=num_envs, 
    horizon=10,
    lambda_=0.1, 
    device="cuda:0", 
    u_max=torch.tensor([3.0, 3.0]),
    u_min=torch.tensor([-3.0, -3.0]),
    step_dependent_dynamics=True,
    terminal_state_cost=terminal_state_cost
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

            # Compute optimal action and send to real simulator
            action = mppi.command(s)
            conn.sendall(data_transfer.torch_to_bytes(action))