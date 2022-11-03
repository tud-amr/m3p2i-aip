from isaacgym import gymapi
from isaacgym import gymtorch
import torch
from utils import env_conf, sim_init

# Decide if you want a viewer or headless
allow_viewer = True
gym, sim, viewer = sim_init.config_gym(allow_viewer)

## Adding Albert
# Desired number of environments and spacing
num_envs = 12
spacing = 10.0
#Init pose
robot_init_pose = gymapi.Transform()
robot_init_pose.p = gymapi.Vec3(0.0, 0.0, 0.05) 
robot_asset = env_conf.load_albert(gym, sim)

# Create the arena(s) with robots
env_conf.create_robot_arena(gym, sim, num_envs, spacing, robot_asset, robot_init_pose)

gym.prepare_sim(sim)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
num_dofs = gym.get_sim_dof_count(sim)
print('Number of DOFs:', num_dofs) # num_envs * 13

# time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0
step = 0

# set velocity targets
max_vel = 5
zero_vel = torch.zeros(num_dofs, dtype=torch.float32, device="cuda:0")
joint_1 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
joint_2 = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
joint_3 = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
joint_4 = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
joint_5 = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
joint_6 = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
joint_7 = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
joint_8 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
joint_9 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
left_vel = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, -max_vel, max_vel, -max_vel, max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
down_vel = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, -max_vel, -max_vel, -max_vel, -max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
up_vel = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, max_vel, max_vel, max_vel, max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
right_vel = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, max_vel, -max_vel, max_vel, -max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
vel_targets = {"up":up_vel, "down":down_vel, "left":left_vel, "right":right_vel, 
                "1":joint_1, "2":joint_2, "3":joint_3, "4":joint_4, "5":joint_5,
                "6":joint_6, "7":joint_7, "8":joint_8, "9":joint_9}

while viewer is None or not gym.query_viewer_has_closed(viewer):
    sim_init.step(gym, sim)
    step += 1

    # apply action
    for evt in gym.query_viewer_action_events(viewer):
        if evt.value > 0:
            gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(vel_targets[evt.action]))
        else:
            gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(zero_vel))

    sim_init.step_rendering(gym, sim, viewer)
    next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

sim_init.destroy_sim(gym, sim, viewer)