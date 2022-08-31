
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from pytorch_mppi import mppi
from utils import env_conf, sim_init

# Decide if you want a viewer or headless
allow_viewer = True
gym, sim, viewer = sim_init.config_gym(allow_viewer)

## Adding Point robot
num_envs = 90
spacing = 10.0

#Init pose
robot_init_pose = gymapi.Transform()
robot_init_pose.p = gymapi.Vec3(0.0, 0.0, 0.05) 
robot_asset = env_conf.load_point_robot(gym, sim)

# Create the arena(s) with robots
env_conf.create_robot_arena(gym, sim, num_envs, spacing, robot_asset, robot_init_pose)

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 6, 8), gymapi.Vec3(1.5, 0, 0))
gym.prepare_sim(sim)

# get dof state tensor

# This variable is linked to the simulator and will update automatically when refresh is called. Used in the dynamics function of the mppi. 
state =  gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))

# This variable doens't update automatically and is used to save the real (without internal mppi rollout).
real_state = torch.clone(state).view(-1, 4)

def mppi_dynamics(input_state, action):
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(input_state))
    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(action))
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_dof_state_tensor(sim)

    res = torch.clone(state).view(-1, 4)
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
    # noise_sigma = torch.tensor(10, device=d, dtype=torch.float32),
    noise_sigma = torch.tensor([[10, 0], [0, 10]], device="cuda:0", dtype=torch.float32),
    num_samples=num_envs, 
    horizon=20,
    lambda_=1., 
    device="cuda:0", 
    u_max=torch.tensor([3.0, 3.0]),
    u_min=torch.tensor([-3.0, -3.0]),
    step_dependent_dynamics=False,
    terminal_state_cost=terminal_state_cost
    )

step = 0
while viewer is None or not gym.query_viewer_has_closed(viewer):

    # Take saved real_state in correct format for mppi.
    s = real_state.view(-1, 4)[0] # [x, v_x, y, v_y]

    # Compute mppi action. This will internally use the simulator to rollout the dynamics.
    action = mppi.command(s)

    # Repeat the action, since the simulator is configured for all the envs.
    action = action.repeat(num_envs)

    # Reset the simulator to saved real_state
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(real_state))

    # Apply real action. (same action for all envs).
    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(action))
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_dof_state_tensor(sim)

    # Update real_state
    real_state = torch.clone(state).view(-1, 4)
    
    if viewer is not None:
        # Step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

    step+=1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
