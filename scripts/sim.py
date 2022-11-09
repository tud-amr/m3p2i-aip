from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from pytorch_mppi import mppi
from utils import env_conf, sim_init
import time
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

# time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0

import socket
import time
import io
server_address = './uds_socket'

def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()

def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)
    

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.connect(server_address)
    while viewer is None or not gym.query_viewer_has_closed(viewer):
        # Take saved real_state in correct format for mppi.
        # s = saved_dof_state.view(-1, 4)[0] # [x, v_x, y, v_y]

        # Compute mppi action. This will internally use the simulator to rollout the dynamics.
        s.sendall(torch_to_bytes(dof_states))
        data = s.recv(1024)

        s.sendall(torch_to_bytes(root_states))
        b = s.recv(1024)
        data = bytes_to_torch(b)

        # Apply real action. (same action for all envs).
        gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(data))

        # Step the similation
        sim_init.step(gym, sim)
        sim_init.refresh_states(gym, sim)

        # Step rendering
        sim_init.step_rendering(gym, sim, viewer)
        next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

# Destroy the simulation
sim_init.destroy_sim(gym, sim, viewer)