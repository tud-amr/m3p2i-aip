from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from utils import env_conf, sim_init

# Decide if you want a viewer or headless
allow_viewer = True
gym, sim, viewer = sim_init.config_gym(allow_viewer)

## Adding Point robot
# Desired number of environments and spacing
num_envs = 4
spacing = 10.0
#Init pose
robot_init_pose = gymapi.Transform()
robot_init_pose.p = gymapi.Vec3(0.0, 0.0, 0.05) 
robot_asset = env_conf.load_point_robot(gym, sim)

# Create the arena(s) with robots
control_type = "vel_control"
env_conf.create_robot_arena(gym, sim, num_envs, spacing, robot_asset, robot_init_pose, control_type)
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 6, 8), gymapi.Vec3(1.5, 0, 0))

gym.prepare_sim(sim)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
num_dofs = gym.get_sim_dof_count(sim)
num_actors = gym.get_sim_actor_count(sim)

# acquire root state tensor descriptor and wrap it in a PyTorch Tensor
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
root_tensor = gymtorch.wrap_tensor(_root_tensor)
saved_root_tensor = root_tensor.clone()

# get relevant info
print("root_tensor", root_tensor.size())
print('number of DOFs:', num_dofs) # num_envs * dof_per_actor
print("dof_state size:", dof_states.size()) # [num_dofs, 2]
print("pos", dof_states[:,0])
print("vel", dof_states[:,1])
print("actor num", num_actors)

# time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0
step = 0

# set velocity targets
zero_vel = torch.zeros(num_dofs, dtype=torch.float32, device="cuda:0")
up_vel = torch.tensor([-2, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
down_vel = torch.tensor([2, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
left_vel = torch.tensor([0, 2], dtype=torch.float32, device="cuda:0").repeat(num_envs)
right_vel = torch.tensor([0, -2], dtype=torch.float32, device="cuda:0").repeat(num_envs)
vel_targets = {"up":up_vel, "down":down_vel, "left":left_vel, "right":right_vel}
actions = torch.zeros(num_dofs, dtype=torch.float32, device="cuda:0")

while viewer is None or not gym.query_viewer_has_closed(viewer):
    sim_init.step(gym, sim)
    step += 1

    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    for evt in gym.query_viewer_action_events(viewer):
        if evt.value > 0:
            if control_type == "pos_control":
                current_pos = dof_states[:,0]
                # current_pos += vel_targets[evt.action] # contiguous error
                # gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(current_pos))
                print("current pos", current_pos)
                gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(current_pos+vel_targets[evt.action]))
            if control_type == "vel_control":
                ## 1. normal version for all the robots
                gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(vel_targets[evt.action]))

                ## 2. specify which robot using slicing 
                # robot_id = 2
                # actions[robot_id*2 : robot_id*2+2] = torch.tensor([-2, 0], dtype=torch.float32, device="cuda:0")
                # gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(actions))

                ## 3. specify which robot using indexed method (not working)
                # actor_indices = torch.tensor([1, 2], dtype=torch.int32, device="cuda:0")  
                # up_vel = torch.tensor([-2, 0], dtype=torch.float32, device="cuda:0")
                # gym.set_dof_velocity_target_tensor_indexed(sim, gymtorch.unwrap_tensor(actions), gymtorch.unwrap_tensor(actor_indices), 2)
            if control_type == "force_control":
                gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(vel_targets[evt.action]))
        else:
            if control_type == "pos_control":
                pass
            if control_type == "vel_control":
                gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(zero_vel))
            if control_type == "force_control":
                gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(zero_vel))

    # print(gymtorch.wrap_tensor(_root_tensor))
    # gym.set_actor_root_state_tensor(sim, _root_tensor) ## gymtorch.unwrap_tensor(saved_root_tensor) not working

    sim_init.step_rendering(gym, sim, viewer)
    next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

sim_init.destroy_sim(gym, sim, viewer)