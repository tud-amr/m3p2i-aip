from isaacgym import gymapi
import math 
import torch
import numpy as np
from pathlib import Path
import utils.path_utils as path_utils

box1_pose = gymapi.Transform()
box1_pose.p = gymapi.Vec3(1, 2, 0)

box2_pose = gymapi.Transform()
box2_pose.p = gymapi.Vec3(-1, 1, 0)         # for push, navigation
# box2_pose.p = gymapi.Vec3(2, -2, 0)       # for reactive push
# box2_pose.p = gymapi.Vec3(3.75, -3.75, 0) # at corner, pull first

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

block_goal = torch.tensor([3, -3], dtype=torch.float32, device="cuda:0")
goal3_pose = gymapi.Transform()
goal3_pose.p = gymapi.Vec3(block_goal[0], block_goal[1], 0)

corner1_pose = gymapi.Transform()
corner1_pose.p = gymapi.Vec3(-3.75, -3.75, 0)

corner2_pose = gymapi.Transform()
corner2_pose.p = gymapi.Vec3(3.75, -3.75, 0)

corner3_pose = gymapi.Transform()
corner3_pose.p = gymapi.Vec3(-3.75, 3.75, 0)

corner4_pose = gymapi.Transform()
corner4_pose.p = gymapi.Vec3(3.75, 3.75, 0)

# docking_station_loc = torch.tensor([0, -3], dtype=torch.float32, device="cuda:0") # in the middle (for aif)
docking_station_loc = torch.tensor([-3.75, -3.75], dtype=torch.float32, device="cuda:0") # close to corner
recharge_pose = gymapi.Transform()
recharge_pose.p = gymapi.Vec3(docking_station_loc[0], docking_station_loc[1], 0)

color_vec_box1 = gymapi.Vec3(0.5, 0.1, 0.7)
color_vec_box2 = gymapi.Vec3(0.2, 0.1, 0.2)
color_vec_crate = gymapi.Vec3(4/255, 160/255, 218/255)
color_vec_box3 = gymapi.Vec3(0.5, 0.1, 0.3)

color_vec_fixed = gymapi.Vec3(0.8, 0.2, 0.2)
color_vec_walls= gymapi.Vec3(0.1, 0.1, 0.1)
color_vec_recharge= gymapi.Vec3(0.3, 1, 1)
color_vec_dyn_obs = gymapi.Vec3(0.8, 0.2, 0.2)
color_vec_corner = gymapi.Vec3(0.0, 1, 0)

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

def add_arena(sim, gym, env, environment_type, origin_x, origin_y, index):
    wall_pose = gymapi.Transform()
    color_vec_walls= gymapi.Vec3(0.1, 0.1, 0.1)
    if environment_type in ["normal", "battery"]:
        square_size = 8
        wall_thickness = 0.1
    if environment_type == "lab":
        square_size = 5
        wall_thickness = 0.05
    # Add 4 walls
    wall_pose.p = gymapi.Vec3(square_size/2+origin_x, origin_y, 0.0)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)
    add_box(sim, gym, env, wall_thickness, square_size, 0.2, wall_pose, color_vec_walls, True, "wall1", index)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1)
    wall_pose.p = gymapi.Vec3(-square_size/2+origin_x, origin_y, 0.0)
    add_box(sim, gym, env, wall_thickness, square_size, 0.2, wall_pose, color_vec_walls, True, "wall2", index)
    wall_pose.p = gymapi.Vec3(origin_x, square_size/2+origin_y, 0.0)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    add_box(sim, gym, env, wall_thickness, square_size, 0.2, wall_pose, color_vec_walls, True, "wall3", index)
    wall_pose.p = gymapi.Vec3(origin_x, -square_size/2+origin_y, 0.0)
    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    add_box(sim, gym, env, wall_thickness, square_size, 0.2, wall_pose, color_vec_walls, True, "wall4", index)

def load_robot(robot, gym, sim):
    if robot == "albert":
        robot_asset = load_albert(gym, sim)
    elif robot == "boxer":
        robot_asset = load_boxer(gym, sim)
    elif robot == "point_robot":
        robot_asset = load_point_robot(gym, sim)
    elif robot == "panda":
        robot_asset = load_panda(gym, sim)
    elif robot == "husky":
        robot_asset = load_husky(gym, sim)
    elif robot == "heijn":
        robot_asset = load_heijn(gym, sim)
    else:
        print("Invalid robot type")
    return robot_asset

def load_albert(gym, sim):
    # Load asset
    asset_root = path_utils.get_assets_path()
    point_robot_asset_file = "urdf/albert/albert.urdf"
    print("Loading asset '%s' from '%s'" % (point_robot_asset_file, asset_root))
    asset_options = gymapi.AssetOptions()
    # asset_options.fix_base_link = True
    # asset_options.disable_gravity = True
    asset_options.armature = 0.01
    robot_asset = gym.load_asset(sim, asset_root, point_robot_asset_file, asset_options)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.01)    
    return robot_asset

def load_boxer(gym, sim):
    # Load asset
    asset_root = path_utils.get_assets_path()
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
    asset_root = path_utils.get_assets_path()
    asset_file = "urdf/pointRobot.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.armature = 0.01
    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return robot_asset

def load_heijn(gym, sim):
    # Load robot asset
    asset_root = path_utils.get_assets_path()
    asset_file = "urdf/heijn.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.armature = 0.01
    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return robot_asset

def load_panda(gym, sim):
    # Load asset
    asset_root = path_utils.get_assets_path()
    franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = True
    asset_options.disable_gravity = True
    asset_options.armature = 0.01
    print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
    franka_asset = gym.load_asset(
        sim, asset_root, franka_asset_file, asset_options)
    return franka_asset

def load_husky(gym, sim):
    # Load asset
    asset_root = path_utils.get_assets_path()
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
        test_cornor = False
        obstacle_handle = add_box(sim, gym, env, 0.3, 0.4, 0.5, obstacle_pose, color_vec_fixed, True, "obstacle", index)
        dyn_obs_handle = add_box(sim, gym, env,0.4, 0.4, 0.1, dyn_obs_pose, color_vec_dyn_obs, False, "dyn_obs", index)

        box1_handle = add_box(sim, gym, env,0.4, 0.4, 0.1, box1_pose, color_vec_box1, False, "box1", index)
        box2_handle = add_box(sim, gym, env,0.4, 0.4, 0.1, box2_pose, color_vec_box2, False, "box2", index)
        box3_handle = add_box(sim, gym, env,0.4, 0.4, 0.1, box3_pose, color_vec_box3, False, "box3", index)

        if not test_cornor:
            # goal_region1 = add_box(sim, gym, env, 1, 1, 0.01, goal1_pose, color_vec_box1, True, "goal_region1", -2) # No collisions with goal region
            # goal_region2 = add_box(sim, gym, env, 1, 1, 0.01, goal2_pose, color_vec_box2, True, "goal_region2", -2) # No collisions with goal region
            # goal_region3 = add_box(sim, gym, env, 1, 1, 0.01, goal3_pose, color_vec_corner, True, "goal_region3", -2) # No collisions with goal region

            recharge_region = add_box(sim, gym, env, 0.45, 0.45, 0.01, recharge_pose, color_vec_corner, True, "goal_region", -2) # No collisions with recharge region
        # add movable squar box
        y_axis = add_box(sim, gym, env, 0.05, 0.5, 0.01, yaxis_pose, gymapi.Vec3(0.0, 1, 0.2), True, "y", -2)
        x_axis = add_box(sim, gym, env, 0.5, 0.05, 0.01, xaxis_pose, gymapi.Vec3(1, 0.0, 0.2), True, "x", -2)       

        if test_cornor:
            corner_region1 = add_box(sim, gym, env, 0.45, 0.45, 0.01, corner1_pose, color_vec_corner, True, "corner_region1", -2)
            corner_region2 = add_box(sim, gym, env, 0.45, 0.45, 0.01, corner2_pose, color_vec_corner, True, "corner_region2", -2)
            corner_region3 = add_box(sim, gym, env, 0.45, 0.45, 0.01, corner3_pose, color_vec_corner, True, "corner_region3", -2)
            corner_region4 = add_box(sim, gym, env, 0.45, 0.45, 0.01, corner4_pose, color_vec_corner, True, "corner_region4", -2)
        
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

def add_panda_arena(gym, sim, env, robot_asset, i):
    # Create table asset
    table_pos = [0.0, 0.0, 1.0]
    table_thickness = 0.05
    table_opts = gymapi.AssetOptions()
    table_opts.fix_base_link = True
    table_asset = gym.create_box(sim, *[1.2, 1.2, table_thickness], table_opts)
    
    # Create table stand asset
    table_stand_height = 0.1
    table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
    table_stand_opts = gymapi.AssetOptions()
    table_stand_opts.fix_base_link = True
    table_stand_asset = gym.create_box(sim, *[0.2, 0.2, table_stand_height], table_stand_opts)

    # Create shelf asset 
    shelf_stand_height = 0.3
    shelf_stand_pos = [0.5, 0.0, 1.0 + table_thickness / 2 + shelf_stand_height / 2]
    shelf_stand_opts = gymapi.AssetOptions()
    shelf_stand_opts.fix_base_link = True
    shelf_stand_asset = gym.create_box(sim, *[0.2, 0.2, shelf_stand_height], shelf_stand_opts)

    # Create obstacle asset
    obs_height = 0.02
    obs_pos = [0.35, 0.0, 1.7 + table_thickness / 2 + obs_height / 2]
    obs_opts = gymapi.AssetOptions()
    obs_opts.disable_gravity = True
    obs_asset = gym.create_box(sim, *[0.2, 0.2, obs_height], obs_opts)

    # Create cubeA asset
    cubeA_opts = gymapi.AssetOptions()
    cubeA_size = 0.050
    cubeA_asset = gym.create_box(sim, *([cubeA_size] * 3), cubeA_opts)
    cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

    # Create cubeB asset
    cubeB_opts = gymapi.AssetOptions()
    cubeB_size = 0.050
    cubeB_asset = gym.create_box(sim, *([cubeB_size] * 3), cubeB_opts)
    cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)

    # Define start pose for panda
    panda_start_pose = gymapi.Transform()
    panda_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
    panda_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # Define start pose for table
    table_start_pose = gymapi.Transform()
    table_start_pose.p = gymapi.Vec3(*table_pos)
    table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
    # self.reward_settings["table_height"] = self._table_surface_pos[2]

    # Define start pose for table stand
    table_stand_start_pose = gymapi.Transform()
    table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
    table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # Define start pose for shelf stand
    shelf_stand_start_pose = gymapi.Transform()
    shelf_stand_start_pose.p = gymapi.Vec3(*shelf_stand_pos)
    shelf_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    
    # Define start pose for obs
    obs_start_pose = gymapi.Transform()
    obs_start_pose.p = gymapi.Vec3(*obs_pos)
    obs_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # Define start pose for cubes
    cubeA_start_pose = gymapi.Transform()
    cubeA_start_pose.p = gymapi.Vec3(0.2, -0.2, 1.05) # on the table
    # cubeA_start_pose.p = gymapi.Vec3(0.45, 0, 1.35) # on the shelf # 0.42
    cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    cubeB_start_pose = gymapi.Transform()
    cubeB_start_pose.p = gymapi.Vec3(0.2, 0.2, 1.06)
    cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # Create panda robot
    panda_actor = gym.create_actor(env, robot_asset, panda_start_pose, "panda", i, 0, 0)

    # Create table
    table_actor = gym.create_actor(env, table_asset, table_start_pose, "table", i, 1, 0)
    table_stand_actor = gym.create_actor(env, table_stand_asset, table_stand_start_pose, "table_stand",i, 1, 0)
    
    # Create cubes
    cubeA_id = gym.create_actor(env, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
    cubeB_id = gym.create_actor(env, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)
    gym.set_rigid_body_color(env, cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
    gym.set_rigid_body_color(env, cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

    # Create shelf
    shelf_creator = gym.create_actor(env, shelf_stand_asset, shelf_stand_start_pose, "shelf_stand",i, 1, 0)

    # Create obstacle
    obs_creater = gym.create_actor(env, obs_asset, obs_start_pose, "obs",i, 1, 0)

    cubeA_rbid = gym.get_actor_rigid_body_index(env, cubeA_id, 0, gymapi.DOMAIN_SIM) # 13
    cubeB_rbid = gym.get_actor_rigid_body_index(env, cubeB_id, 0, gymapi.DOMAIN_SIM) # 14
    shelf_rbid = gym.get_actor_rigid_body_index(env, shelf_creator, 0, gymapi.DOMAIN_SIM) # 15
    obs_rbid = gym.get_actor_rigid_body_index(env, obs_creater, 0, gymapi.DOMAIN_SIM) # 16

    return panda_actor

def add_albert_arena(gym, sim, env, robot_asset, i):
    # Create table asset
    table_pos = [0.0, 0.0, 0.5]
    table_thickness = 0.05
    table_opts = gymapi.AssetOptions()
    table_opts.fix_base_link = True
    table_asset = gym.create_box(sim, *[1.2, 1.2, table_thickness], table_opts)

    # Create cubeA asset
    cubeA_opts = gymapi.AssetOptions()
    cubeA_size = 0.050
    cubeA_asset = gym.create_box(sim, *([cubeA_size] * 3), cubeA_opts)
    cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

    # Create cubeB asset
    cubeB_opts = gymapi.AssetOptions()
    cubeB_size = 0.050
    cubeB_asset = gym.create_box(sim, *([cubeB_size] * 3), cubeB_opts)
    cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)

    # Define start pose for albert
    albert_start_pose = gymapi.Transform()
    albert_start_pose.p = gymapi.Vec3(0.9, 0, 0) # 2, 2
    albert_start_pose.r = gymapi.Quat(0.0, 0.0, 0, 1)

    # Define start pose for table
    table_start_pose = gymapi.Transform()
    table_start_pose.p = gymapi.Vec3(*table_pos)
    table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
    # self.reward_settings["table_height"] = self._table_surface_pos[2]


    # Define start pose for cubes
    cubeA_start_pose = gymapi.Transform()
    cubeA_start_pose.p = gymapi.Vec3(0.5, -0.3, 0.55)
    cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    cubeB_start_pose = gymapi.Transform()
    cubeB_start_pose.p = gymapi.Vec3(0.5, 0.2, 0.56)
    cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # Create table
    table_actor = gym.create_actor(env, table_asset, table_start_pose, "table", i, 1, 0)

    # Create cubes
    cubeA_id = gym.create_actor(env, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
    cubeB_id = gym.create_actor(env, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)
    gym.set_rigid_body_color(env, cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
    gym.set_rigid_body_color(env, cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

    # Create panda robot
    albert_actor = gym.create_actor(env, robot_asset, albert_start_pose, "albert", i, 0, 0)

    cubeA_rbid = gym.get_actor_rigid_body_index(env, cubeA_id, 0, gymapi.DOMAIN_SIM) 
    cubeB_rbid = gym.get_actor_rigid_body_index(env, cubeB_id, 0, gymapi.DOMAIN_SIM) 

    return albert_actor


def get_default_franka_state(gym, robot_asset):

    franka_dof_props = gym.get_asset_dof_properties(robot_asset)
    franka_lower_limits = franka_dof_props["lower"]
    franka_upper_limits = franka_dof_props["upper"]
    franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
    
    # default dof states and position targets
    franka_num_dofs = gym.get_asset_dof_count(robot_asset)
    # franka_mids[7:] = franka_upper_limits[7:] # grippers open
    default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = franka_mids[:]
    default_dof_state["pos"][3] = -2

    return default_dof_state
                        
def create_robot_arena(gym, sim, num_envs, spacing, robot_asset, pose, viewer, environment_type, control_type = "vel_control"):
    # Some common handles for later use
    envs = []
    robot_handles = []
    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))
    if environment_type == 'cube':
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0, 1.5, 2.8), gymapi.Vec3(0, 0, 1))
    elif environment_type == 'albert_arena':
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0, 4, 2.8), gymapi.Vec3(0, 0, 1))
    else:
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 6, 8), gymapi.Vec3(1.5, 0, 0))

    for i in range(num_envs):
        # Create env
        env = gym.create_env(sim, gymapi.Vec3(-spacing, 0.0, -spacing), gymapi.Vec3(spacing, spacing, spacing), num_per_row)
        envs.append(env)

        if environment_type in ["normal" ,"battery", "lab"]:
            add_arena(sim, gym, env, environment_type, 0, 0, i) # origin_x, origin_y, index
            add_obstacles(sim, gym, env, environment_type, index = i)
            robot_handle = gym.create_actor(env, robot_asset, pose, "robot", i, 1)
            if environment_type == "battery":
                gym.set_rigid_body_color(env, robot_handle, -1, gymapi.MESH_VISUAL_AND_COLLISION, color_vec_battery_ok)
        else:
            if environment_type == "cube":
                robot_handle = add_panda_arena(gym, sim, env, robot_asset, i)
            elif environment_type == "albert_arena":
                robot_handle = add_albert_arena(gym, sim, env, robot_asset, i)
            if 'default_dof_state' not in locals():
                default_dof_state = get_default_franka_state(gym, robot_asset)
            gym.set_actor_dof_states(env, robot_handle, default_dof_state, gymapi.STATE_ALL)

        robot_handles.append(robot_handle)

        # Update point bot dynamics / control mode
        props = gym.get_asset_dof_properties(robot_asset)
        if environment_type == "cube":
            # Set panda dof properties
            props["driveMode"][7:].fill(gymapi.DOF_MODE_VEL)
            props["stiffness"][7:].fill(800.0)
            props["damping"][7:].fill(40.0)
        elif control_type == "pos_control":
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

        albert_rigid_body_names = ['base_link', 'chassis_link', 'internal_link', 'lift_link', 'top_mount_bottom', 'extrusion1', 'top_mount', 'mmrobot_link0', 'mmrobot_link1', 'mmrobot_link2', 'mmrobot_link3', 'mmrobot_link4', 'mmrobot_link5', 'mmrobot_link6', 'mmrobot_link7', 'mmrobot_link8', 'mmrobot_hand', 'mmrobot_leftfinger', 'mmrobot_rightfinger', 'extrusion2', 'extrusion3', 'extrusion4', 'rotacastor_left_link', 'rotacastor_right_link', 'wheel_left_link', 'wheel_right_link']
        if gym.get_asset_rigid_body_names(robot_asset) == albert_rigid_body_names:
            # Rigid bodies and rigid shapes are different. A rigid bodies can have 0 to n rigid shapes
            shape_props = gym.get_actor_rigid_shape_properties(env, robot_handle)
            shape_props[20].friction = 0.
            shape_props[20].torsion_friction = 0.
            shape_props[20].rolling_friction = 0.
            shape_props[21].friction = 0.
            shape_props[21].torsion_friction = 0.
            shape_props[21].rolling_friction = 0.
            gym.set_actor_rigid_shape_properties(env, robot_handle, shape_props)

    return envs, robot_handles
