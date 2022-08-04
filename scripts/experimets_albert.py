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

# time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0

# MPPI settings
step = 0
mppi_step_count = 100

# sample initial action sequence
action_sequence = (1 - -1) * torch.rand(mppi_step_count, num_dofs, device="cuda:0") -1

while viewer is None or not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    step += 1

    # apply sampled action
    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(action_sequence[step % mppi_step_count]))
    
    if step % mppi_step_count == 0:
        # reset states
        reset_states = torch.zeros(2, num_dofs, dtype=torch.float32, device="cuda:0")
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(reset_states))

        # sample action sequence (random between -1, 1)
        action_sequence = 2 * torch.rand(mppi_step_count, num_dofs, device="cuda:0") - 1
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
