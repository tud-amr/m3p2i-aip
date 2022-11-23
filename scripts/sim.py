from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from pytorch_mppi import mppi
from utils import env_conf, sim_init
import time
import numpy as np
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

# Decide if you want a viewer or headless
allow_viewer = True
visualize_rollouts = True

## Adding Point robot
num_envs = 1 
spacing = 10.0
dt = 0.1

robot = "boxer"               # choose from "point_robot", "boxer", "albert"
environment_type = "normal"            # choose from "normal", "battery"
control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
gym, sim, viewer, envs, robot_handles = sim_init.make(allow_viewer, num_envs, spacing, robot, environment_type, control_type)

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 6, 8), gymapi.Vec3(1.5, 0, 0))
gym.prepare_sim(sim)

# Init simulation tensors and torch wrappers (see /docs/programming/tensors.html)
dof_state =  gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
actor_root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
gym.refresh_actor_root_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)

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
    buff.seek(0)
    return torch.load(buff)
    

t4 = time.monotonic()
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.connect(server_address)
    while viewer is None or not gym.query_viewer_has_closed(viewer):
        t1 = time.monotonic()
        # Compute mppi action. This will internally use the simulator to rollout the dynamics.
        s.sendall(torch_to_bytes(dof_state))
        data = s.recv(1024)

        s.sendall(torch_to_bytes(actor_root_state))
        b = s.recv(2048)
        actions = bytes_to_torch(b)
        t2 = time.monotonic()

        # Clear lines at the beginning
        gym.clear_lines(viewer)

        # Send message and receive rollout states
        if visualize_rollouts:
            s.sendall(b"Visualize rollouts")
            rollouts = s.recv(2**18)
            rollouts = bytes_to_torch(rollouts)

            height = 0.2
            line_length = rollouts.size()[1] - 1
            blue = np.array([[0, 0, 255]], dtype='float32').repeat(line_length, axis=0)
            green = np.array([[0, 255, 0]], dtype='float32').repeat(line_length, axis=0)
            for i, traj in enumerate(rollouts):
                line_array = torch.cat((
                    torch.cat((traj[:-1], torch.ones(line_length, 1, device=torch.device('cuda:0'))*height), axis=1),
                    torch.cat((traj[1:], torch.ones(line_length, 1, device=torch.device('cuda:0'))*height), axis=1)),
                    axis=1
                )
                if i == 0:
                    gym.add_lines(viewer, envs[0], line_length, line_array.cpu().numpy(), blue)
                else:
                    gym.add_lines(viewer, envs[0], line_length, line_array.cpu().numpy(), green)

        action = actions[0]
        if robot == 'boxer':
            r = 0.08
            L = 2*0.157
            # Diff drive fk
            action_fk = action.clone()
            action_fk[0] = (action[0] / r) - ((L*action[1])/(2*r))
            action_fk[1] = (action[0] / r) + ((L*action[1])/(2*r))
            action = action_fk

        # Apply real action. (same action for all envs).
        gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(action))
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)

        t3 = time.monotonic()

        if viewer is not None:
            # Step rendering
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)

            sleep_time = dt - (t3-t1)
            if sleep_time > 0:
                time.sleep(sleep_time)

        t_prev = t4
        t4 = time.monotonic()
        print(f"FPS: {1/(t4-t_prev):.2f}, MPPI: {1e3*(t2-t1):.0f} ms, Simulation: {1e3*(t3-t2):.0f} ms, Graphics: {1e3*(t4-t3):.0f} ms")


gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
