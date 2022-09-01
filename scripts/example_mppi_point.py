
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
num_envs = 500
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

# save copies of states, in order to reset the rollout of MPPI
saved_dof_state = dof_state.clone().view(-1, 4)
saved_actor_root_state = actor_root_state.clone()

def mppi_dynamics(input_state, action, t):
    if t == 0:    
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(input_state))
    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(action))
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_dof_state_tensor(sim)

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

# time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0

while viewer is None or not gym.query_viewer_has_closed(viewer):

    # Take saved real_state in correct format for mppi.
    s = saved_dof_state.view(-1, 4)[0] # [x, v_x, y, v_y]

    # Compute mppi action. This will internally use the simulator to rollout the dynamics.
    action = mppi.command(s)
    all_actions = torch.zeros(num_envs * 2, device="cuda:0")
    all_actions[:2] = action

    # Reset the simulator to saves
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(saved_dof_state))
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(saved_actor_root_state))

    # Apply real action. (same action for all envs).
    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(all_actions))
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)

    # Update saves
    saved_dof_state = torch.clone(dof_state).view(-1, 4)
    saved_actor_root_state = torch.clone(actor_root_state)
    
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
