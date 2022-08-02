from hashlib import new
import math
from isaacgym import gymapi
import numpy as np
from decision_making import ai_agent, adaptive_action_selection
import isaac_int_req_templates, isaac_state_action_templates 
from utils import env_conf

# Initialize gym
gym = gymapi.acquire_gym()

# get default set of parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 100
sim_params.substeps = 1
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
asset_root = "../assets"
point_robot_asset_file = "urdf/pointRobot.urdf"
print("Loading asset '%s' from '%s'" % (point_robot_asset_file, asset_root))
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.armature = 0.01
point_robot_asset = gym.load_asset(sim, asset_root, point_robot_asset_file, asset_options)

# Set up the env grid
num_envs = 1
spacing = 6.0
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

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)
    # create arena
    env_conf.add_arena(sim, gym, env, 8,0.1, 0, 0, i) # Wall size, wall thickness, origin_x, origin_y, index

    # add movable squar box
    movable_obstacle_handle = env_conf.add_box(sim, gym, env,0.2, 0.2, 0.2, env_conf.movable_box_pose, env_conf.color_vec_movable, False, "movable_box", i)
    
    # add fixed obstacle
    obstacle_handle = env_conf.add_box(sim, gym, env, 0.3, 0.4, 0.5, env_conf.obstacle_pose, env_conf.color_vec_fixed, True, "obstacle", i)

    goal_region = env_conf.add_box(sim, gym, env, 1, 1, 0.01, env_conf.goal_pose, env_conf.color_vec_goal, True, "goal_region", -2) # No collisions with goal region
    recharge_region = env_conf.add_box(sim, gym, env,1 , 1, 0.01, env_conf.recharge_pose, env_conf.color_vec_recharge, True, "goal_region", -2) # No collisions with recharge region
    
    # add point robot
    point_robot_handle = gym.create_actor(env, point_robot_asset, pose, "pointRobot", i, -1)
    point_robot_handles.append(point_robot_handle)

    gym.set_rigid_body_color(env, point_robot_handle, -1, gymapi.MESH_VISUAL_AND_COLLISION, env_conf.color_vec_movable)
    num_bodies = gym.get_actor_rigid_body_count(env, point_robot_handles[-1])

props = gym.get_asset_dof_properties(point_robot_asset)
print('\nSome properties of the dof', props)
props["driveMode"].fill(gymapi.DOF_MODE_VEL)
props["stiffness"].fill(0.0)
props["damping"].fill(600.0)

# Controller tuning
Kp = 1

def apply_control(u):
    # Get robot position at current time
    dof_states = gym.get_actor_dof_states(env, point_robot_handle, gymapi.STATE_ALL)
    pos = dof_states["pos"]   # all positions   

    vel_target = [0, 0]
    # u = 1: go normal, u = 1: slow-down, u = 2: go_recharge
    if u == 1: 
        # (TODO) set penalty for high velocities
        pass
    if u == 'go_recharge': 
        # (TODO) set penalty for high velocities
        # Simple P controller, to be substituted
        vel_target = Kp*[env_conf.docking_station_loc[0] - pos[0], env_conf.docking_station_loc[1] - pos[1]]

    for i in range(num_envs):
        gym.set_actor_dof_properties(envs[i], point_robot_handles[i], props)
        gym.set_actor_dof_velocity_targets(envs[i], point_robot_handles[i], vel_target)

#Inspection states (you can also set them or get the states of the simulation instead of the reduced info contained in the DOFs)
dof_states = gym.get_actor_dof_states(env, point_robot_handle, gymapi.STATE_ALL)
pos = dof_states["pos"]   # all positions
vel = dof_states["vel"]   # all velocities

# Point camera at environments
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
cam_pos = gymapi.Vec3(0.0, 10.0, 10.0)
cam_target = gymapi.Vec3(0.0, 0.0, -1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

def battery_sim(battery_level):
    if np.linalg.norm(pos - env_conf.docking_station_loc) < 0.5:
         new_level = battery_level + 0.1
    else:
        new_level = battery_level - 0.1        # We can make this proportional to the velocity + a low dischrge factor
    if new_level < 0:
        new_level = 0
    if new_level > 100:
        new_level = 100
    return new_level

def get_battery_obs(battery_level):
    if battery_level > 55: 
        obs_battery = 0  # Battery is ok for active inference
        for n in range(num_bodies):
            gym.set_rigid_body_color(env, point_robot_handles[-1], n, gymapi.MESH_VISUAL, env_conf.color_vec_battery_ok)
    elif battery_level > 35:
        for n in range(num_bodies):
            gym.set_rigid_body_color(env, point_robot_handles[-1], n, gymapi.MESH_VISUAL, env_conf.color_vec_battery_low)
        obs_battery = 1  # Battery is low
    else:
        obs_battery = 2  # Battery is critical
        for n in range(num_bodies):
            gym.set_rigid_body_color(env, point_robot_handles[-1], n, gymapi.MESH_VISUAL, env_conf.color_vec_battery_critical)
    return obs_battery

###############################################################################
###############################################################################

# Active inference agent
## Initialization
# ----------------- 
# Define the required mdp structures 
mdp_battery = isaac_int_req_templates.MDPBattery() 
# Define ai agent with related mdp structure to reason about
ai_agent_internal = ai_agent.AiAgent(mdp_battery)

mdp_isAt = isaac_state_action_templates.MDPIsAt() 
# Define ai agent with related mdp structure to reason about
ai_agent_task = ai_agent.AiAgent(mdp_isAt)

## Decision making
#-------------------
# A typical sequence for decision making, ideally this should be repeated at a certain frequency

# Set the preference for the battery 
ai_agent_internal.set_preferences(np.array([[1.], [0], [0]])) # Fixed preference for battery ok, following ['ok', 'low', 'critcal'] 
ai_agent_task.set_preferences(np.array([[1.], [0]])) # Fixed preference for battery ok, following ['at_goal', 'not_at_goal']

t_decision = 0
battery_level = 100

while not gym.query_viewer_has_closed(viewer):
    # t = gym.get_sim_time(sim)
    # Prepare observations and change robot color according to battery level
    battery_level = battery_sim(battery_level)

    # Decision making loop every second
    if t_decision == 0 or t_decision > 100:
        o_battery = get_battery_obs(battery_level)
        
        #print('battery level', battery_level)
        #print('Battey observation', o_battery)
        o_isAt = 1
        
        outcome_task, curr_action_task = adaptive_action_selection.adapt_act_sel(ai_agent_task, o_isAt)
        outcome_internal, curr_action_internal = adaptive_action_selection.adapt_act_sel(ai_agent_internal, o_battery)

        t_decision = 0

        # Printouts
        #print('The selected action from the task is', curr_action_task)
        print('The selected action from the internal requirements is', curr_action_internal)

        #print('The battery state is:',  ai_agent_internal._mdp.state_names[np.argmax(ai_agent_internal.get_current_state())])
        # print('The action is:', ai_agent_internal._mdp.action_names[u])
        # print('Measured battery level', battery_level)

        # print('The isAt state is:',  ai_agent_task._mdp.state_names[np.argmax(ai_agent_task.get_current_state())])
        # print('The action is:', ai_agent_task._mdp.action_names[ut])
        if curr_action_internal is not 'idle_success':
            print('Free energy', sum(ai_agent_task.G))
    # TODO: now we have the two actions one for the task and one for the internal needs, with related expected free energies. When do we do what? 

    # Compute and apply control action accoridng to action selection outcome (TODO) Substitute with MPPI
    apply_control(curr_action_internal)
    
    t_decision = t_decision + 1
    
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

