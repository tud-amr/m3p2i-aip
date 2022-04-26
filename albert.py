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
point_robot_asset_file = "urdf/albert/albert.urdf"


print("Loading asset '%s' from '%s'" % (point_robot_asset_file, asset_root))
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.armature = 0.01
point_robot_asset = gym.load_asset(sim, asset_root, point_robot_asset_file, asset_options)

# Set up the env grid
num_envs = 1
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
# Some common handles for later use
envs = []
point_robot_handles = []

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))

# To add an actor to an environment, you must specify the desired pose,
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.01)

# Obstacles
pose_box = gymapi.Transform()
pose_box.p = gymapi.Vec3(0.5, 0.3, 0.0)

# Additional assets from API
asset_options_objects = gymapi.AssetOptions()
asset_options_objects.fix_base_link = False

width = 0.1
height = 0.2
depth = 0.1

box_asset = gym.create_box(sim, width, height, depth, asset_options_objects)
box_asset = gym.create_box(sim, width, height, depth, asset_options_objects)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add point robot
    point_robot_handle = gym.create_actor(env, point_robot_asset, pose, "pointRobot", i, -1)
    point_robot_handles.append(point_robot_handle)

    # Add obstacles
    box_handle = gym.create_actor(env, box_asset, pose_box, "Box", i, -1)


# Get infor about one of the n environments
num_bodies = gym.get_actor_rigid_body_count(env, point_robot_handle)
num_joints = gym.get_actor_joint_count(env, point_robot_handle)
num_dofs = gym.get_actor_dof_count(env, point_robot_handle)

# Inspecting the pointRobot
print('\nThe point robot has:\n')
print('Number of rigid bodies:', num_bodies)
print('Number of joints:', num_joints)
print('Number of DOFs:', num_dofs)

props = gym.get_asset_dof_properties(point_robot_asset)
print('\nSome properties of the dof', props)
props["driveMode"].fill(gymapi.DOF_MODE_VEL)
props["stiffness"].fill(0.0)
props["damping"].fill(600.0)

print('\nAnd after changing the properties:', props)

# Controlling
# rotocasters needs to have the sae velovity as the wheel to go straight (last 4 parameters)
vel_targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]
for i in range(num_envs):
    gym.set_actor_dof_properties(envs[i], point_robot_handles[i], props)
    gym.set_actor_dof_velocity_targets(envs[i], point_robot_handles[i], vel_targets)
# Inspection states (you can also set them or get the states of the simulation instead of the reduced info contained in the DOFs)
dof_states = gym.get_actor_dof_states(env, point_robot_handle, gymapi.STATE_ALL)
pos = dof_states["pos"]   # all positions
vel = dof_states["vel"]   # all velocities

print('This is the current position and velocity, zero at start', [pos, vel])

# Point camera at environments
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
contact = gymapi.RigidContact()

while not gym.query_viewer_has_closed(viewer):
    t = gym.get_sim_time(sim)

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    contact = gym.get_env_rigid_contacts(env)
    print(contact[1])
    #print(gym.get_rigid_linear_velocity(env,movable_obstacle_handle))
    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
