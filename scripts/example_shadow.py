from isaacgym import gymapi
from isaacgym import gymtorch
import torch
from utils import env_conf, sim_init
import numpy as np 

# Decide if you want a viewer or headless
allow_viewer = False

## Adding Point robot
num_envs = 1
spacing = 10.0

robot = "shadow_hand"               # choose from "point_robot", "boxer", "albert"
environment_type = "shadow"         # choose from "arena", "battery"
control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
gym, sim, viewer, envs, robot_handles = sim_init.make(allow_viewer, num_envs, spacing, robot, environment_type, control_type)

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 6, 8), gymapi.Vec3(1.5, 0, 0))
gym.prepare_sim(sim)

# subscribe to input events. This allows input to be used to interact
# with the simulation
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "left")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "down")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "right")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "up")

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
num_dofs = gym.get_sim_dof_count(sim)
print(num_dofs)
# time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0

# MPPI settings
step = 0
mppi_step_count = 100

# sample initial action sequence
action_sequence = (1 - -1) * torch.rand(mppi_step_count, num_dofs, device="cuda:0") -1
curr_vel = torch.zeros(1, num_dofs, dtype=torch.float32, device="cuda:0")
curr_vel[0,0] = 2 
curr_vel[0,1] = -2

all_actions = torch.zeros(1, num_dofs, dtype=torch.float32, device="cuda:0")

_net_cf = gym.acquire_net_contact_force_tensor(sim)
net_cf = gymtorch.wrap_tensor(_net_cf)

max_vel = 1
max_yaw = 3

curr_vel[0,0] = 0
curr_vel[0,1] = 0

actor_root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
gym.refresh_actor_root_state_tensor(sim)

while viewer is None or not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    step += 1

    _net_cf = gym.refresh_net_contact_force_tensor(sim)

    evts = [evt.action for evt in gym.query_viewer_action_events(viewer)]

    gym.refresh_actor_root_state_tensor(sim)
   
    # gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(all_actions))

    # if step % mppi_step_count == 0:
    #     # reset states
    #     reset_states = torch.zeros(2, num_dofs, dtype=torch.float32, device="cuda:0")
    #     gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(reset_states))

    #     # sample action sequence (random between -1, 1)
    #     action_sequence = 2 * torch.rand(mppi_step_count, num_dofs, device="cuda:0") - 1
    if viewer is not None:
        # Step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)
    # Net contact forces with black walls. Indexes are according to how you add the actors in the env
    if torch.max(net_cf[0:4])>1 or torch.max(net_cf[0:4])<-1:
        print("Collision")
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
