import math
from isaacgym import gymapi
import numpy as np

# Initialize gym
gym = gymapi.acquire_gym()

# get default set of parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

# create sim with these parameters
compute_device_id = 0               # index of CUDA-enabled GPU to be used for simulation.    
graphics_device_id = 0              # index of GPU to be used for rendering
physics_engine = gymapi.SIM_PHYSX

sim = gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)

# configure the ground plane
# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

# Load asset
asset_root = "assets"
point_robot_asset_file = "urdf/pointRobot.urdf"
print("Loading asset '%s' from '%s'" % (point_robot_asset_file, asset_root))
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.armature = 0.01
point_robot_asset = gym.load_asset(sim, asset_root, point_robot_asset_file, asset_options)

# Set up the env grid
num_envs = 100
spacing = 5.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
# Some common handles for later use
envs = []
point_robot_handles = []

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))

# To add an actor to an environment, you must specify the desired pose,
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.05)

def add_box(width, height, depth, pose, color, isFixed, name, index):
    # Additional assets from API
    asset_options_objects = gymapi.AssetOptions()
    asset_options_objects.fix_base_link = isFixed

    object_asset = gym.create_box(sim, width, height, depth, asset_options_objects)
    # Add obstacles
    box_handle = gym.create_actor(env, object_asset, pose, name, index, -1)
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    return box_handle

def add_arena(square_size, wall_thikhness, origin_x, origin_y, index):
    wall_pose = gymapi.Transform()
    # Add 4 walls
    wall_pose.p = gymapi.Vec3(square_size/2+origin_x, origin_y, 0.0)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)
    add_box(wall_thikhness, square_size, 1, wall_pose, color_vec_walls, True, "wall1", index)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)
    wall_pose.p = gymapi.Vec3(-square_size/2+origin_x, origin_y, 0.0)
    add_box(wall_thikhness, square_size, 1, wall_pose, color_vec_walls, True, "wall2", index)
    wall_pose.p = gymapi.Vec3(origin_x, square_size/2+origin_y, 0.0)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    add_box(wall_thikhness, square_size, 1, wall_pose, color_vec_walls, True, "wall3", index)
    wall_pose.p = gymapi.Vec3(origin_x, -square_size/2+origin_y, 0.0)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    add_box(wall_thikhness, square_size, 1, wall_pose, color_vec_walls, True, "wall4", index)

movable_box_pose = gymapi.Transform()
movable_box_pose.p = gymapi.Vec3(0.5, 0.5, 0)

obstacle_pose = gymapi.Transform()
obstacle_pose.p = gymapi.Vec3(1, 1, 0)

color_vec_movable = gymapi.Vec3(0.8, 0.2, 0.2)
color_vec_fixed = gymapi.Vec3(0.3, 0.7, 0.7)
color_vec_walls= gymapi.Vec3(0.1, 0.1, 0.1)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # create arena
    add_arena(4,0.1, 0, 0, i) # Wall size, wall thickness, origin_x, origin_y, index

    # add movable squar box
    movable_obstacle_handle = add_box(0.2, 0.2, 0.2, movable_box_pose, color_vec_movable, False, "movable_box", i)
    
    # add fixed obstacle
    obstacle_handle = add_box(0.3, 0.4, 0.5, obstacle_pose, color_vec_fixed, True, "obstacle", i)

    # add point robot
    point_robot_handle = gym.create_actor(env, point_robot_asset, pose, "pointRobot", i, 1)
    point_robot_handles.append(point_robot_handle)

# Controlling
props = gym.get_asset_dof_properties(point_robot_asset)
print('\nSome properties of the dof', props)
props["driveMode"].fill(gymapi.DOF_MODE_VEL)
props["stiffness"].fill(0.0)
props["damping"].fill(600.0)
vel_targets = [0.5, 0.5]
for i in range(num_envs):
    gym.set_actor_dof_properties(envs[i], point_robot_handles[i], props)
    gym.set_actor_dof_velocity_targets(envs[i], point_robot_handles[i], vel_targets)
# Inspection states (you can also set them or get the states of the simulation instead of the reduced info contained in the DOFs)
dof_states = gym.get_actor_dof_states(env, point_robot_handle, gymapi.STATE_ALL)
pos = dof_states["pos"]   # all positions
vel = dof_states["vel"]   # all velocities

# Point camera at environments
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

while not gym.query_viewer_has_closed(viewer):
    t = gym.get_sim_time(sim)

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
