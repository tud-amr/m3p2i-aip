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
allow_viewer = True
gym, sim, viewer = sim_init.config_gym(allow_viewer)

## Adding Point robot
num_envs = 300
spacing = 10.0

#Init pose
robot_init_pose = gymapi.Transform()
robot_init_pose.p = gymapi.Vec3(0.0, 0.0, 0.05) 
robot_asset = env_conf.load_point_robot(gym, sim)

# Create the arena(s) with robots
envs = env_conf.create_robot_arena(gym, sim, num_envs, spacing, robot_asset, robot_init_pose)

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 6, 8), gymapi.Vec3(1.5, 0, 0))
gym.prepare_sim(sim)

# Init simulation tensors and torch wrappers (see /docs/programming/tensors.html)
dof_state =  gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
actor_root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
gym.refresh_actor_root_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)

def mppi_dynamics(input_state, action, t):
    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(action))
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_dof_state_tensor(sim)

    if allow_viewer:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

    res = torch.clone(dof_state).view(-1, 4)
    return res

def running_cost(state, action):
    return torch.zeros(num_envs, device="cuda:0")

    # return cost

def terminal_state_cost(states, actions):
    states = torch.cat((states[0, :, -1, 0].unsqueeze(1), states[0, :, -1, 2].unsqueeze(1)), 1)
    dist = torch.linalg.norm(states - torch.tensor([3, -3], device="cuda:0"), axis=1)
    return dist**2

mppi = mppi.MPPI(
    dynamics=mppi_dynamics, 
    running_cost=running_cost, 
    nx=4, 
    noise_sigma = torch.tensor([[10, 0], [0, 10]], device="cuda:0", dtype=torch.float32),
    num_samples=num_envs, 
    horizon=20,
    lambda_=1., 
    device="cuda:0", 
    u_max=torch.tensor([3.0, 3.0]),
    u_min=torch.tensor([-3.0, -3.0]),
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
    buff.seek(0)
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
            s = _dof_state.view(-1, 4) # [x, v_x, y, v_y]
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(s))
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(_actor_root_state))

            gym.refresh_actor_root_state_tensor(sim)
            gym.refresh_dof_state_tensor(sim)

            action = mppi.command(s)

            conn.sendall(torch_to_bytes(action))
