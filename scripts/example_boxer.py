from isaacgym import gymapi
from isaacgym import gymtorch
import torch
from utils import env_conf, sim_init
import numpy as np 

# Decide if you want a viewer or headless
allow_viewer = True
gym, sim, viewer = sim_init.config_gym(allow_viewer)

## Adding Boxer
# Desired number of environments and spacing
num_envs = 128
spacing = 10.0
#Init pose
robot_init_pose = gymapi.Transform()
robot_init_pose.p = gymapi.Vec3(0.0, 0.0, 0.05) 
robot_asset = env_conf.load_boxer(gym, sim)

# Create the arena(s) with robots
env_conf.create_robot_arena(gym, sim, num_envs, spacing, robot_asset, robot_init_pose)
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 6, 8), gymapi.Vec3(1.5, 0, 0))

gym.prepare_sim(sim)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
num_dofs = gym.get_sim_dof_count(sim)

# time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0

# MPPI settings
step = 0
mppi_step_count = 100

# sample initial action sequence
action_sequence = (1 - -1) * torch.rand(mppi_step_count, num_dofs, device="cuda:0") -1
zero_vel = torch.zeros(1, num_dofs, dtype=torch.float32, device="cuda:0")

_net_cf = gym.acquire_net_contact_force_tensor(sim)
net_cf = gymtorch.wrap_tensor(_net_cf)

max_vel = 5
left_vel = torch.tensor([-max_vel, max_vel, -max_vel, max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
down_vel = torch.tensor([-max_vel, -max_vel, -max_vel, -max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
up_vel = torch.tensor([max_vel, max_vel, max_vel, max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
right_vel = torch.tensor([max_vel, -max_vel, max_vel, -max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
vel_targets = {"up":up_vel, "down":down_vel, "left":left_vel, "right":right_vel}

while viewer is None or not gym.query_viewer_has_closed(viewer):
    sim_init.step(gym, sim)
    step += 1

    _net_cf = gym.refresh_net_contact_force_tensor(sim)

    for evt in gym.query_viewer_action_events(viewer):
        if evt.value > 0:
            gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(vel_targets[evt.action]))
        else:
            gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(zero_vel))

    # if step % mppi_step_count == 0:
    #     # reset states
    #     reset_states = torch.zeros(2, num_dofs, dtype=torch.float32, device="cuda:0")
    #     gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(reset_states))

    #     # sample action sequence (random between -1, 1)
    #     action_sequence = 2 * torch.rand(mppi_step_count, num_dofs, device="cuda:0") - 1
    
    # Net contact forces with black walls. Indexes are according to how you add the actors in the env
    if torch.max(net_cf[0:4])>1 or torch.max(net_cf[0:4])<-1:
        print("Collision")
    else:
        print("Ok")
        
    sim_init.step_rendering(gym, sim, viewer)
    next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

sim_init.destroy_sim(gym, sim, viewer)
