from isaacgym import gymapi
import math 
import torch
import numpy as np
from pathlib import Path

box1_pose = gymapi.Transform()
box1_pose.p = gymapi.Vec3(1, -2, 0)

box2_pose = gymapi.Transform()
box2_pose.p = gymapi.Vec3(-2, 1, 0)

box3_pose = gymapi.Transform()
box3_pose.p = gymapi.Vec3(3, 0, 0)

crate_pose = gymapi.Transform()
crate_pose.p = gymapi.Vec3(-1, -1, 0)

obstacle_pose = gymapi.Transform()
obstacle_pose.p = gymapi.Vec3(2, 2, 0)

dyn_obs_pose = gymapi.Transform()
dyn_obs_pose.p = gymapi.Vec3(-2, 2, 0)

goal1_pose = gymapi.Transform()
goal1_pose.p = gymapi.Vec3(-3, 3, 0)

goal2_pose = gymapi.Transform()
goal2_pose.p = gymapi.Vec3(3, 3, 0)

goal3_pose = gymapi.Transform()
goal3_pose.p = gymapi.Vec3(3, -3, 0)

docking_station_loc = torch.tensor([-3, -3], dtype=torch.float32, device="cuda:0")
recharge_pose = gymapi.Transform()
recharge_pose.p = gymapi.Vec3(docking_station_loc[1], docking_station_loc[0], 0)

color_vec_box1 = gymapi.Vec3(0.5, 0.1, 0.7)
color_vec_box2 = gymapi.Vec3(0.2, 0.1, 0.2)
color_vec_crate = gymapi.Vec3(4/255, 160/255, 218/255)
color_vec_box3 = gymapi.Vec3(0.5, 0.1, 0.3)

color_vec_fixed = gymapi.Vec3(0.8, 0.2, 0.2)
color_vec_walls= gymapi.Vec3(0.1, 0.1, 0.1)
color_vec_recharge= gymapi.Vec3(0.0, 0.9, 0.3)
color_vec_dyn_obs = gymapi.Vec3(0.8, 0.2, 0.2)

color_vec_battery_ok = gymapi.Vec3(0.0, 0.7, 0.5)
color_vec_battery_low = gymapi.Vec3(0.8, 0.5, 0.)
color_vec_battery_critical = gymapi.Vec3(0.8, 0.2, 0.2)

xaxis_pose = gymapi.Transform()
xaxis_pose.p = gymapi.Vec3(0.25, 0, 0)

yaxis_pose = gymapi.Transform()
yaxis_pose.p = gymapi.Vec3(0, 0.25, 0.01)

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
    add_box(sim, gym, env, wall_thikhness, square_size, 0.2, wall_pose, color_vec_walls, True, "wall1", index)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)
    wall_pose.p = gymapi.Vec3(-square_size/2+origin_x, origin_y, 0.0)
    add_box(sim, gym, env, wall_thikhness, square_size, 0.2, wall_pose, color_vec_walls, True, "wall2", index)
    wall_pose.p = gymapi.Vec3(origin_x, square_size/2+origin_y, 0.0)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    add_box(sim, gym, env, wall_thikhness, square_size, 0.2, wall_pose, color_vec_walls, True, "wall3", index)
    wall_pose.p = gymapi.Vec3(origin_x, -square_size/2+origin_y, 0.0)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    add_box(sim, gym, env, wall_thikhness, square_size, 0.2, wall_pose, color_vec_walls, True, "wall4", index)

def load_robot(robot, gym, sim):
    if robot == "albert":
        robot_asset = load_albert(gym, sim)
    elif robot == "boxer":
        robot_asset = load_boxer(gym, sim)
    elif robot == "point_robot":
        robot_asset = load_point_robot(gym, sim)
    elif robot == "franka":
        robot_asset = load_franka(gym, sim)
    elif robot == "husky":
        robot_asset = load_husky(gym, sim)
    elif robot == "heijn":
        robot_asset = load_heijn(gym, sim)
    else:
        print("Invalid robot type")
    return robot_asset

def get_asset_path():
    curr_path = Path().absolute()
    list = str(curr_path).split('/')
    asset_path = ''
    for i in list:
        if i != "scripts":
            asset_path += i + "/"
        else:
            break
    asset_path += 'assets'
    return asset_path

def load_albert(gym, sim):
    # Load asset
    asset_root = get_asset_path()
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
    asset_root = get_asset_path()
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
    asset_root = get_asset_path()
    asset_file = "urdf/pointRobot.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.armature = 0.01
    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return robot_asset

def load_heijn(gym, sim):
    # Load robot asset
    asset_root = get_asset_path()
    asset_file = "urdf/heijn.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.armature = 0.01
    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return robot_asset

def load_franka(gym, sim):
    # Load asset
    asset_root = get_asset_path()
    franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = True
    asset_options.armature = 0.01

    print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
    franka_asset = gym.load_asset(
        sim, asset_root, franka_asset_file, asset_options)
    return franka_asset

def load_husky(gym, sim):
    # Load asset
    asset_root = get_asset_path()
    husky_asset_file = "urdf/husky_description/husky_rectangular_bumper.urdf"

    print("Loading asset '%s' from '%s'" % (husky_asset_file, asset_root))
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.armature = 0.01

    # set asset options for the husky
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.use_mesh_materials = True
    asset_options.flip_visual_attachments = False
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.disable_gravity = False
    # for overriding bad values in urdf
    asset_options.override_com = True
    asset_options.override_inertia = True
    asset_options.vhacd_enabled = True
    # more accurate shape collisions
    asset_options.vhacd_params.resolution = 300000
    asset_options.vhacd_params.max_convex_hulls = 10
    asset_options.vhacd_params.max_num_vertices_per_ch = 64

    robot_asset = gym.load_asset(sim, asset_root, husky_asset_file, asset_options)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.01)    
    return robot_asset

def add_obstacles(sim, gym, env, environment_type, index):
    if environment_type == "normal":
        # add fixed obstacle
        obstacle_handle = add_box(sim, gym, env, 0.3, 0.4, 0.5, obstacle_pose, color_vec_fixed, True, "obstacle", index)
        dyn_obs_handle = add_box(sim, gym, env,0.4, 0.4, 0.1, dyn_obs_pose, color_vec_dyn_obs, False, "dyn_obs", index)

        box1_handle = add_box(sim, gym, env,0.4, 0.4, 0.1, box1_pose, color_vec_box1, False, "box1", index)
        box2_handle = add_box(sim, gym, env,0.4, 0.4, 0.1, box2_pose, color_vec_box2, False, "box2", index)
        box3_handle = add_box(sim, gym, env,0.4, 0.4, 0.1, box3_pose, color_vec_box3, False, "box3", index)

        goal_region1 = add_box(sim, gym, env, 1, 1, 0.01, goal1_pose, color_vec_box1, True, "goal_region1", -2) # No collisions with goal region
        goal_region2 = add_box(sim, gym, env, 1, 1, 0.01, goal2_pose, color_vec_box3, True, "goal_region2", -2) # No collisions with goal region
        goal_region3 = add_box(sim, gym, env, 1, 1, 0.01, goal3_pose, color_vec_box3, True, "goal_region3", -2) # No collisions with goal region

        recharge_region = add_box(sim, gym, env,1 , 1, 0.01, recharge_pose, color_vec_recharge, True, "goal_region", -2) # No collisions with recharge region
        # add movable squar box
        y_axis = add_box(sim, gym, env, 0.05, 0.5, 0.01, yaxis_pose, gymapi.Vec3(0.0, 1, 0.2), True, "y", -2)
        x_axis = add_box(sim, gym, env, 0.5, 0.05, 0.01, xaxis_pose, gymapi.Vec3(1, 0.0, 0.2), True, "x", -2)        
        
    elif environment_type == "battery":
        # add fixed obstacle
        obstacle_handle = add_box(sim, gym, env, 0.3, 0.4, 0.5, obstacle_pose, color_vec_fixed, True, "obstacle", index)
        movable_obstacle_handle = add_box(sim, gym, env,0.2, 0.2, 0.2, box1_pose, color_vec_box1, False, "movable_box", index)

        goal_region = add_box(sim, gym, env, 1, 1, 0.01, goal1_pose, color_vec_box1, True, "goal_region", -2) # No collisions with goal region
        
        recharge_region = add_box(sim, gym, env,1 , 1, 0.01, recharge_pose, color_vec_recharge, True, "goal_region", -2) # No collisions with recharge region
        
        y_axis = add_box(sim, gym, env, 0.05, 0.5, 0.01, yaxis_pose, gymapi.Vec3(0.0, 1, 0.2), True, "y", -2)
        x_axis = add_box(sim, gym, env, 0.5, 0.05, 0.01, xaxis_pose, gymapi.Vec3(1, 0.0, 0.2), True, "x", -2)

    elif environment_type == "lab":        
        crate_handle = add_box(sim, gym, env,0.5, 0.3, 0.3, crate_pose, color_vec_crate, False, "box2", index)
        y_axis = add_box(sim, gym, env, 0.05, 0.5, 0.01, yaxis_pose, gymapi.Vec3(0.0, 1, 0.2), True, "y", -2)
        x_axis = add_box(sim, gym, env, 0.5, 0.05, 0.01, xaxis_pose, gymapi.Vec3(1, 0.0, 0.2), True, "x", -2)        

        # Set AH crate mass 
        crate_props = gym.get_actor_rigid_body_properties(env, crate_handle)
        crate_props[0].mass = 1. # Set 1kg mass
        gym.set_actor_rigid_body_properties(env, crate_handle, crate_props)      
    else:
        print("Invalid environment type")
    
def create_robot_arena(gym, sim, num_envs, spacing, robot_asset, pose, viewer, environment_type, control_type = "vel_control"):
    # Some common handles for later use
    envs = []
    robot_handles = []
    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 6, 8), gymapi.Vec3(1.5, 0, 0))
    for i in range(num_envs):
        # Create env
        env = gym.create_env(sim, gymapi.Vec3(-spacing, 0.0, -spacing), gymapi.Vec3(spacing, spacing, spacing), num_per_row)
        envs.append(env)
        
        if environment_type == "normal" or environment_type == "battery":
            wall_size = 8
            wall_thickness = 0.1
        if environment_type == "lab":
            wall_size = 5
            wall_thickness = 0.05

        add_arena(sim, gym, env, wall_size, wall_thickness, 0, 0, i) # Wall size, wall thickness, origin_x, origin_y, index
        
        # Add obstacles
        add_obstacles(sim, gym, env, environment_type, index = i)

        # Add robot
        robot_handle = gym.create_actor(env, robot_asset, pose, "robot", i, 1)
        robot_handles.append(robot_handle)
        if environment_type == "battery":
            gym.set_rigid_body_color(env, robot_handle, -1, gymapi.MESH_VISUAL_AND_COLLISION, color_vec_battery_ok)

        # Update point bot dynamics / control mode
        props = gym.get_asset_dof_properties(robot_asset)
        if control_type == "pos_control":
            props["driveMode"].fill(gymapi.DOF_MODE_POS)
            props["stiffness"].fill(1000.0)
            props["damping"].fill(200.0)
        elif control_type == "vel_control":
            props["driveMode"].fill(gymapi.DOF_MODE_VEL)
            props["stiffness"].fill(0.0)        # The stiffness parameter should be set to zero.
            props["damping"].fill(600.0)        # The torques applied by the PD controller will be proportional to the damping parameter
        elif control_type == "force_control":
            props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
            props["stiffness"].fill(0.0)
            props["damping"].fill(0.0)
        else:
            print("Invalid control type!")
        gym.set_actor_dof_properties(env, robot_handle, props)

        # Set friction of rotacasters to zero for boxer
        boxer_rigid_body_names = ['base_link_ori', 'base_link', 'chassis_link', 'rotacastor_left_link', 'rotacastor_right_link', 'wheel_left_link', 'wheel_right_link', 'ee_link']
        if gym.get_asset_rigid_body_names(robot_asset) == boxer_rigid_body_names:
            shape_props = gym.get_actor_rigid_shape_properties(env, robot_handle)
            shape_props[1].friction = 0.
            shape_props[1].torsion_friction = 0.
            shape_props[1].rolling_friction = 0.
            shape_props[2].friction = 0.
            shape_props[2].torsion_friction = 0.
            shape_props[2].rolling_friction = 0.
            gym.set_actor_rigid_shape_properties(env, robot_handle, shape_props)

    return envs, robot_handles
