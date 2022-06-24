from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch

import math
import random

# parse arguments
args = gymutil.parse_arguments(description="Simple Panda Example")
args.use_gpu = True
# args.graphics_device_id = -1

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 2.0 / 100.0
sim_params.substeps = 1
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.use_gpu_pipeline = args.use_gpu
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

# creating sim
gym = gymapi.acquire_gym()
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# optionally create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# viewer = None

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0
gym.add_ground(sim, plane_params)

# Load robot asset
asset_root = "../assets"
asset_file = "urdf/pointRobot.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.armature = 0.01

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Set up the env grid
num_envs = 9 
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Some common handles for later use
envs = []
robot_handles = []
print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.05)
pose.r = gymapi.Quat(0, 0.0, 0.0, 1.0)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add point bot
    robot_handle = gym.create_actor(env, robot_asset, pose, "point", i, 1)
    robot_handles.append(robot_handle)

    # update point bot dynamics / control mode
    props = gym.get_asset_dof_properties(robot_asset)
    props["driveMode"].fill(gymapi.DOF_MODE_VEL)
    props["stiffness"].fill(0.0)
    props["damping"].fill(600.0)
    gym.set_actor_dof_properties(env, robot_handle, props)

gym.prepare_sim(sim)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
num_dofs = gym.get_sim_dof_count(sim)

# time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0

# mppi settings
step = 0
mppi_step_count = 100
# sample initial action sequence
action_sequence = (1 - -1) * torch.rand(mppi_step_count, num_dofs, device="cuda:0") - 1

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
