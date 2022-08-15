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
# Only working for 1 environment
num_envs = 1
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
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
num_dofs = gym.get_sim_dof_count(sim)

state =  gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
print("state", state)

# time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0

# MPPI settings
step = 0
mppi_step_count = 100

# subscribe to input events. This allows input to be used to interact
# with the simulation
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "left")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "down")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "right")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "up")


# sample initial action sequence
action_sequence = (1 - -1) * torch.rand(mppi_step_count, num_dofs, device="cuda:0") - 1
zero_vel = torch.zeros(1, num_dofs, dtype=torch.float32, device="cuda:0")
up_vel = torch.tensor([-2, 0], dtype=torch.float32, device="cuda:0")
down_vel = torch.tensor([2, 0], dtype=torch.float32, device="cuda:0")
left_vel = torch.tensor([0, 2], dtype=torch.float32, device="cuda:0")
right_vel = torch.tensor([0, -2], dtype=torch.float32, device="cuda:0")

# Test for applying exxternal forces
# forces = torch.zeros((num_envs, 15, 3), device="cuda:0", dtype=torch.float)
# pos = torch.zeros((num_envs, 15, 3), device="cuda:0", dtype=torch.float)
#forces[:, 4, 0] = 100
#forces[:, 4, 1] = 150

while viewer is None or not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    step += 1

    # Test for applying exxternal forces
    #gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE)
    #gym.apply_rigid_body_force_at_pos_tensors(sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(pos), gymapi.ENV_SPACE)

    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "left" and evt.value > 0:
            gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(left_vel))
        elif evt.action == "down" and evt.value > 0:
            gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(down_vel))
        elif evt.action == "up" and evt.value > 0:
            gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(up_vel))
        elif evt.action == "right" and evt.value > 0:
            gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(right_vel))
        else:
            gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(zero_vel))
    
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

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
