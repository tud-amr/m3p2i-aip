

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

## Adding Point robot
num_envs = 2000
spacing = 10.0

robot = "point_robot"               # choose from "point_robot", "boxer", "albert"
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

# Save copies of states, in order to reset the rollout of MPPI
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
    # State: for each environment, the current state containing position and velocity
    # Action: same but for control input
    
    state_pos = torch.cat((state[:, 0].unsqueeze(1), state[:, 2].unsqueeze(1)), 1)
    task_cost = torch.linalg.norm(state_pos - torch.tensor([3, 3], device="cuda:0"), axis=1)
    
    w_u = 0.01 # Weight for control input, more dominant when close to the goal
    control_cost = torch.sum(torch.square(action),1)
    
    # Contact forces
    _net_cf = gym.acquire_net_contact_force_tensor(sim)
    net_cf = gymtorch.wrap_tensor(_net_cf)
    _net_cf = gym.refresh_net_contact_force_tensor(sim)
    # Take only forces in x,y in modulus for each environment. Avoid all collisions
    net_cf = torch.sum(torch.abs(torch.cat((net_cf[:, 0].unsqueeze(1), net_cf[:, 1].unsqueeze(1)), 1)),1)
    coll_cost = torch.sum(net_cf.reshape([num_envs, int(net_cf.size(dim=0)/num_envs)]), 1)
    w_c = 100000 # Weight for collisions
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
    noise_sigma = torch.tensor([[3, 0], [0, 3]], device="cuda:0", dtype=torch.float32),
    num_samples=num_envs, 
    horizon=10,
    lambda_=0.1, 
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