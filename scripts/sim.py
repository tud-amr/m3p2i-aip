
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from pytorch_mppi import mppi
from utils import env_conf, sim_init
import time
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

# Decide if you want a viewer or headless
allow_viewer = True
gym, sim, viewer = sim_init.config_gym(allow_viewer)

## Adding Point robot
num_envs = 1 
spacing = 10.0

#Init pose
robot_init_pose = gymapi.Transform()
robot_init_pose.p = gymapi.Vec3(0.0, 0.0, 0.05) 
# robot_asset = env_conf.load_point_robot(gym, sim)
robot_asset = env_conf.load_boxer(gym, sim)

# Create the arena(s) with robots
envs = env_conf.create_robot_arena(gym, sim, num_envs, spacing, robot_asset, robot_init_pose)

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
    return torch.load(buff)
    

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.connect(server_address)
    while viewer is None or not gym.query_viewer_has_closed(viewer):
        # Take saved real_state in correct format for mppi.
        # s = saved_dof_state.view(-1, 4)[0] # [x, v_x, y, v_y]

        # Compute mppi action. This will internally use the simulator to rollout the dynamics.
        s.sendall(torch_to_bytes(dof_state))
        data = s.recv(1024)

        s.sendall(torch_to_bytes(actor_root_state))
        b = s.recv(1024)
        action = bytes_to_torch(b)

        r = 0.08
        L = 2*0.157
        action_fk = action.clone()
        action_fk[0] = (action[0] / r) - ((L*action[1])/(2*r))
        action_fk[1] = (action[0] / r) + ((L*action[1])/(2*r))

        # Apply real action. (same action for all envs).
        gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(action_fk))
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)

        if viewer is not None:
            # Step rendering
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)


        # time logging
        t = gym.get_elapsed_time(sim)
        if t >= next_fps_report:
            t2 = gym.get_elapsed_time(sim)
            fps = frame_count / (t2 - t1)
            print("FPS %.1f (%.1f)" % (fps, fps * num_envs))
            frame_count = 0
            t1 = gym.get_elapsed_time(sim)
            next_fps_report = t1 + 2.0
        frame_count += 1


gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

