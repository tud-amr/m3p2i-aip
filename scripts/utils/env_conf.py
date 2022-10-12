from isaacgym import gymapi
import math 
import numpy as np

box1_pose = gymapi.Transform()
box1_pose.p = gymapi.Vec3(1, -2, 0)

box2_pose = gymapi.Transform()
box2_pose.p = gymapi.Vec3(-1, 1, 0)

box3_pose = gymapi.Transform()
box3_pose.p = gymapi.Vec3(3, 0, 0)

obstacle_pose = gymapi.Transform()
obstacle_pose.p = gymapi.Vec3(2, 2, 0)

goal1_pose = gymapi.Transform()
goal1_pose.p = gymapi.Vec3(-3, 3, 0)

goal2_pose = gymapi.Transform()
goal2_pose.p = gymapi.Vec3(3, 3, 0)

goal3_pose = gymapi.Transform()
goal3_pose.p = gymapi.Vec3(3, -3, 0)

docking_station_loc = [-3, -3]
recharge_pose = gymapi.Transform()
recharge_pose.p = gymapi.Vec3(docking_station_loc[1], docking_station_loc[0], 0)

color_vec_box1 = gymapi.Vec3(0.5, 0.1, 0.7)
color_vec_box2 = gymapi.Vec3(0.2, 0.1, 0.2)
color_vec_box3 = gymapi.Vec3(0.5, 0.1, 0.3)

color_vec_fixed = gymapi.Vec3(0.8, 0.2, 0.2)
color_vec_walls= gymapi.Vec3(0.1, 0.1, 0.1)
color_vec_recharge= gymapi.Vec3(0.0, 0.9, 0.3)

color_vec_battery_ok = gymapi.Vec3(0.0, 0.7, 0.5)
color_vec_battery_low = gymapi.Vec3(0.8, 0.5, 0.)
color_vec_battery_critical = gymapi.Vec3(0.8, 0.2, 0.2)

def add_box(sim, gym, env, width, height, depth, pose, color, isFixed, name, index):
    # Additional assets from API
    asset_options_objects = gymapi.AssetOptions()
    asset_options_objects.fix_base_link = isFixed

    object_asset = gym.create_box(sim, width, height, depth, asset_options_objects)
    # Add obstacles
    box_handle = gym.create_actor(env, object_asset, pose, name, index, -1)
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    return box_handle

def add_arena(sim, gym, env, square_size, wall_thikhness, origin_x, origin_y, index):
    wall_pose = gymapi.Transform()
    color_vec_walls= gymapi.Vec3(0.1, 0.1, 0.1)
    # Add 4 walls
    wall_pose.p = gymapi.Vec3(square_size/2+origin_x, origin_y, 0.0)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)
    add_box(sim, gym, env, wall_thikhness, square_size, 1, wall_pose, color_vec_walls, True, "wall1", index)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)
    wall_pose.p = gymapi.Vec3(-square_size/2+origin_x, origin_y, 0.0)
    add_box(sim, gym, env, wall_thikhness, square_size, 1, wall_pose, color_vec_walls, True, "wall2", index)
    wall_pose.p = gymapi.Vec3(origin_x, square_size/2+origin_y, 0.0)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    add_box(sim, gym, env, wall_thikhness, square_size, 1, wall_pose, color_vec_walls, True, "wall3", index)
    wall_pose.p = gymapi.Vec3(origin_x, -square_size/2+origin_y, 0.0)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    add_box(sim, gym, env, wall_thikhness, square_size, 1, wall_pose, color_vec_walls, True, "wall4", index)

def load_albert(gym, sim):
    # Load asset
    asset_root = "../assets"
    point_robot_asset_file = "urdf/albert/albert.urdf"

    print("Loading asset '%s' from '%s'" % (point_robot_asset_file, asset_root))
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.armature = 0.01
    robot_asset = gym.load_asset(sim, asset_root, point_robot_asset_file, asset_options)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.01)    
    return robot_asset

def load_boxer(gym, sim):
    # Load asset
    asset_root = "../assets"
    point_robot_asset_file = "urdf/boxer/boxer.urdf"

    print("Loading asset '%s' from '%s'" % (point_robot_asset_file, asset_root))
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.armature = 0.01
    robot_asset = gym.load_asset(sim, asset_root, point_robot_asset_file, asset_options)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.01)    
    return robot_asset

def load_point_robot(gym, sim):
    # Load robot asset
    asset_root = "../assets"
    asset_file = "urdf/pointRobot.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.armature = 0.01
    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return robot_asset

def load_franka(gym, sim):
    # Load asset
    asset_root = "../assets"
    franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = True
    asset_options.armature = 0.01

    print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
    franka_asset = gym.load_asset(
        sim, asset_root, franka_asset_file, asset_options)
    return franka_asset

def create_robot_arena(gym, sim, num_envs, spacing, robot_asset, pose, control_type = "vel_control"):
    # Some common handles for later use
    envs = []
    robot_handles = []
    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))
    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, gymapi.Vec3(-spacing, 0.0, -spacing), gymapi.Vec3(spacing, spacing, spacing), num_per_row)
        envs.append(env)
        add_arena(sim, gym, env, 8,0.1, 0, 0, i) # Wall size, wall thickness, origin_x, origin_y, index
        
        # add movable squar box
        box1_handle = add_box(sim, gym, env,0.4, 0.4, 0.1, box1_pose, color_vec_box1, False, "box1", i)
        box2_handle = add_box(sim, gym, env,0.4, 0.4, 0.1, box2_pose, color_vec_box2, False, "box2", i)
        box2_handle = add_box(sim, gym, env,0.4, 0.4, 0.1, box3_pose, color_vec_box3, False, "box3", i)

        # add fixed obstacle
        obstacle_handle = add_box(sim, gym, env, 0.3, 0.4, 0.5, obstacle_pose, color_vec_fixed, True, "obstacle", i)

        goal_region1 = add_box(sim, gym, env, 1, 1, 0.01, goal1_pose, color_vec_box1, True, "goal_region1", -2) # No collisions with goal region
        goal_region2 = add_box(sim, gym, env, 1, 1, 0.01, goal2_pose, color_vec_box2, True, "goal_region2", -2) # No collisions with goal region
        goal_region3 = add_box(sim, gym, env, 1, 1, 0.01, goal3_pose, color_vec_box3, True, "goal_region3", -2) # No collisions with goal region

        recharge_region = add_box(sim, gym, env,1 , 1, 0.01, recharge_pose, color_vec_recharge, True, "goal_region", -2) # No collisions with recharge region
        
        
        # add point bot
        robot_handle = gym.create_actor(env, robot_asset, pose, "robot", i, 1)
        robot_handles.append(robot_handle)

        # update point bot dynamics / control mode
        props = gym.get_asset_dof_properties(robot_asset)
        
        if control_type == "pos_control":
            # print(control_type)
            props["driveMode"].fill(gymapi.DOF_MODE_POS)
            props["stiffness"].fill(1000.0)
            props["damping"].fill(200.0)
        elif control_type == "vel_control":
            # print(control_type)
            props["driveMode"].fill(gymapi.DOF_MODE_VEL)
            props["stiffness"].fill(0.0)        # The stiffness parameter should be set to zero.
            props["damping"].fill(600.0)        # The torques applied by the PD controller will be proportional to the damping parameter
        elif control_type == "force_control":
            # print(control_type)
            props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
            props["stiffness"].fill(0.0)
            props["damping"].fill(0.0)

        gym.set_actor_dof_properties(env, robot_handle, props)