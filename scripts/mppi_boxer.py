from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from pytorch_mppi import mppi
from utils import env_conf, sim_init
import time
import copy
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

# Decide if you want a viewer or headless
allow_viewer = False

## Adding Point robot
num_envs = 300
spacing = 10.0

robot = "boxer"               # choose from "point_robot", "boxer", "albert"
environment_type = "normal"         # choose from "normal", "battery"
control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
gym, sim, viewer, envs, robot_handles = sim_init.make(allow_viewer, num_envs, spacing, robot, environment_type, control_type)

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 6, 8), gymapi.Vec3(1.5, 0, 0))
gym.prepare_sim(sim)

# Init simulation tensors and torch wrappers (see /docs/programming/tensors.html)
dof_state =  gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
actor_root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
gym.refresh_actor_root_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)

root_positions = actor_root_state[:, 0:3]


# Pushing purple blox
block_index = 10 
block_goal = torch.tensor([-3, -3], device="cuda:0")
# nav_goal = torch.tensor([3, 3], device="cuda:0")


def mppi_dynamics(input_state, action, t):

    r = 0.08
    L = 2*0.157
    # Diff drive fk
    actions_fk = action.clone()
    actions_fk[:, 0] = (action[:, 0] / r) - ((L*action[:, 1])/(2*r))
    actions_fk[:, 1] = (action[:, 0] / r) + ((L*action[:, 1])/(2*r))

    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(actions_fk))
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)

    if allow_viewer:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

    res = actor_root_state[12::13]
    res2 = torch.cat([res[:, :2], res[:, 7:9]], axis=1)
    if t==0:
        print(res2[0])
    return res2

def get_push_cost(r_pos):
    block_pos = torch.cat((torch.split(torch.clone(root_positions[:,0:2]), int(torch.clone(root_positions[:,0:2]).size(dim=0)/num_envs))),1)[block_index,:].reshape(num_envs,2)

    robot_to_block = torch.linalg.norm(r_pos - block_pos, axis = 1)
    block_to_goal = torch.linalg.norm(block_goal - block_pos, axis = 1)

    cost = robot_to_block + block_to_goal 

    robot_to_goal = torch.linalg.norm(r_pos - block_goal, axis = 1)
    align_cost = torch.zeros_like(robot_to_goal)
    align_cost[block_to_goal > robot_to_goal] = 1

    cost += align_cost
    return cost

def get_navigation_cost(r_pos):
    return torch.linalg.norm(r_pos - nav_goal, axis=1)


def running_cost(state, action):
    # State: for each environment, the current state containing position and velocity
    # Action: same but for control input
    
    # state_pos = torch.cat((state[:, 0].unsqueeze(1), state[:, 2].unsqueeze(1)), 1)
    control_cost = torch.sum(torch.square(action),1)
    w_u = 0.01
    # Contact forces
    _net_cf = gym.acquire_net_contact_force_tensor(sim)
    net_cf = gymtorch.wrap_tensor(_net_cf)
    _net_cf = gym.refresh_net_contact_force_tensor(sim)
    # Take only forces in x,y in modulus for each environment. Avoid all collisions
    net_cf = torch.sum(torch.abs(torch.cat((net_cf[:, 0].unsqueeze(1), net_cf[:, 1].unsqueeze(1)), 1)),1)
    # The last 6 actors are allowed to collide with eachother (movabable obstacles and robot)
    coll_cost = torch.sum(net_cf.reshape([num_envs, int(net_cf.size(dim=0)/num_envs)])[:,0:9], 1)
    w_c = 10000 # Weight for collisions
    # Binary check for collisions. So far checking all collision with unmovable obstacles. Movable obstacles touching unmovable ones are considered collisions       
    coll_cost[coll_cost>0.1] = 1
    coll_cost[coll_cost<=0.1] = 0
    task_cost = get_push_cost(state[:, :2])
    return  task_cost #+ w_c*coll_cost # + w_u*control_cost 

# def running_cost(state, action):
#     a = torch.linalg.norm(state[:, :2] - torch.tensor([3, 3], device="cuda:0"), axis=1)
#     # print(state[0])
#     # return torch.zeros(num_envs, device="cuda:0")
#     # b = torch.linalg.norm(action, axis=1)
#     return a  
#     # cost = b
#     # return cost
#     # return torch.zeros(num_envs, device="cuda:0")

def terminal_state_cost(states, actions):
    # dist = torch.linalg.norm(states[0, :, -1, :2] - torch.tensor([3, 3], device="cuda:0"), axis=1)
    return torch.zeros(num_envs, device="cuda:0")
    # dist = torch.linalg.norm(states[0, :, -1, :2] - torch.tensor([300, 300], device="cuda:0"), axis=1)
    # return dist

mppi = mppi.MPPI(
    dynamics=mppi_dynamics, 
    running_cost=running_cost, 
    nx=4, 
    noise_sigma = torch.tensor([[10, 0], [0, 10]], device="cuda:0", dtype=torch.float32),
    num_samples=num_envs, 
    horizon=30,
    lambda_=1., 
    device="cuda:0", 
    u_max=torch.tensor([1.5, 5.5]),
    u_min=torch.tensor([-1.5, -5.5]),
    step_dependent_dynamics=True,
    terminal_state_cost=terminal_state_cost
    )

import socket
import time
import os

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
server_address = './uds_socket'

# Make sure the socket does not already exist
try:
    os.unlink(server_address)
except OSError:
    if os.path.exists(server_address):
        raise

import io

def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)  # <--  this is what you were missing
    return buff.read()

def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.bind(server_address)
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        i=0
        while True:
            i+=1
            while True:
                res = conn.recv(1024)
                if not res == b'': break
            r = copy.copy(res)
            _dof_state = bytes_to_torch(r).repeat(num_envs, 1)

            conn.sendall(b"next please")

            while True:
                res = conn.recv(2**14)
                if not res == b'': break
            r = copy.copy(res)
            _actor_root_state = bytes_to_torch(r)
            _actor_root_state = _actor_root_state.repeat(num_envs, 1)

            # Reset the simulator to requested state
            s = _dof_state.view(-1, 4)
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(s))
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(_actor_root_state))

            gym.refresh_actor_root_state_tensor(sim)
            gym.refresh_dof_state_tensor(sim)

            action = mppi.command(actor_root_state)

            conn.sendall(torch_to_bytes(action))
