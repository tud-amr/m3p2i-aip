from isaacgym import gymapi
import math 
import torch
import numpy as np

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

franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0, 0, 0)

table_dims = gymapi.Vec3(0.6, 1.0, 0.108)
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

shelves_dims = gymapi.Vec3(0.6, 1.0, 1.5)
shelf_pose = gymapi.Transform()
shelf_pose.p = gymapi.Vec3(1.85, 3-0.5*shelves_dims.y, 0)

box_size = 0.04
box_pose = gymapi.Transform()
box_pose.p.x = table_pose.p.x + 0.2*np.random.rand()
box_pose.p.y = table_pose.p.y + 0.3*np.random.rand()
box_pose.p.z = table_dims.z + 0.5 * box_size
box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))

product_pose = gymapi.Transform()
product_pose.p.x = table_pose.p.x 
product_pose.p.y = table_pose.p.y 
product_pose.p.z = table_dims.z + 0.08
product_pose.r.x = 0.7068252
product_pose.r.y = 0
product_pose.r.z = 0
product_pose.r.w = 0.7068252

box1_pose_storm = gymapi.Transform()
box1_pose_storm.p = gymapi.Vec3(0.4, 0.2, 0.2)
box2_pose_storm = gymapi.Transform()
box2_pose_storm.p = gymapi.Vec3(0.4, -0.3, 0.25)

envs = []
box_idxs = []
hand_idxs = []
init_pos_list = []
init_rot_list = []

mug_color = gymapi.Vec3(0.85, 0.88, 0.2)

def add_box(sim, gym, env, width, height, depth, pose, color, isFixed, name, index):
    # Additional assets from API
    asset_options_objects = gymapi.AssetOptions()
    asset_options_objects.fix_base_link = isFixed
    object_asset = gym.create_box(sim, width, height, depth, asset_options_objects)
    # Add obstacles
    box_handle = gym.create_actor(env, object_asset, pose, name, index, 0)
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
    elif robot == "panda":
        robot_asset = load_panda(gym, sim)
    elif robot == "panda_no_hand":
        robot_asset = load_panda_no_hand(gym, sim)
    elif robot == "husky":
        robot_asset = load_husky(gym, sim)
    elif robot == "heijn":
        robot_asset = load_heijn(gym, sim)
    elif robot == "omni_panda":
        robot_asset = load_omni_panda(gym, sim)
    elif robot == "shadow_hand":
        robot_asset = load_shadow(gym, sim)
    else:
        print("Invalid robot type")
    return robot_asset

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

def load_heijn(gym, sim):
    # Load robot asset
    asset_root = "../assets"
    asset_file = "urdf/heijn.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.armature = 0.01
    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return robot_asset

def load_panda_no_hand(gym, sim):
    # Load asset
    asset_root = "../assets"
    franka_asset_file = "urdf/franka_description/robots/franka_panda_stick.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

    return franka_asset

def load_panda(gym, sim):
    # Load asset
    asset_root = "../assets"
    franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)
    # configure franka dofs
    franka_dof_props = gym.get_asset_dof_properties(franka_asset)
    franka_lower_limits = franka_dof_props["lower"]
    franka_upper_limits = franka_dof_props["upper"]
    franka_ranges = franka_upper_limits - franka_lower_limits
    franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)
    franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_VEL)
    franka_dof_props["stiffness"][7:].fill(800.0)
    franka_dof_props["damping"][7:].fill(40.0)

    return franka_asset

def load_omni_panda(gym, sim):
    # Load asset
    asset_root = "../assets"
    omni_panda_asset_file = "urdf/omni_panda/omniPandaWithGripper.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    omni_panda_asset = gym.load_asset(sim, asset_root, omni_panda_asset_file, asset_options)
    # configure omni_panda dofs
    omni_panda_dof_props = gym.get_asset_dof_properties(omni_panda_asset)
    omni_panda_dof_props["driveMode"][10:].fill(gymapi.DOF_MODE_VEL)
    omni_panda_dof_props["stiffness"][10:].fill(800.0)
    omni_panda_dof_props["damping"][10:].fill(40.0)

    return omni_panda_asset

def load_husky(gym, sim):
    # Load asset
    asset_root = "../assets"
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

def load_mug(gym, sim):
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.vhacd_enabled = True
    asset_options.vhacd_params = gymapi.VhacdParams()
    asset_options.vhacd_params.resolution = 10

    mug_asset_file = "urdf/objects/mug/mug.urdf"
    mug_asset = gym.load_asset(sim, "../assets", mug_asset_file, asset_options)
    return mug_asset

def load_hageslag(gym, sim):
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.vhacd_enabled = True
    asset_options.vhacd_params = gymapi.VhacdParams()
    asset_options.vhacd_params.resolution = 10

    hageslag_asset_file = "urdf/objects/AH_hagelslag_aruco_0/hageslag_0.urdf"
    hageslag_asset = gym.load_asset(sim, "../assets", hageslag_asset_file, asset_options)
    return hageslag_asset

def load_shelf(gym, sim):
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.vhacd_enabled = True
    asset_options.vhacd_params = gymapi.VhacdParams()
    asset_options.vhacd_params.resolution = 10

    # Testing store from sketchup
    # asset_options.vhacd_params.resolution = 300000
    # asset_options.vhacd_params.max_convex_hulls = 1000
    # asset_options.vhacd_params.max_num_vertices_per_ch = 64
    # shelf_asset_file = "urdf/store/store.urdf"

    shelf_asset_file = "urdf/AH_shelf/shelf.urdf"
    shelf_asset = gym.load_asset(sim, "../assets", shelf_asset_file, asset_options)

    return shelf_asset

def load_shadow(gym, sim):
    asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = False
    asset_options.use_mesh_materials = True
    shadow_asset = gym.load_asset(sim, "../assets", asset_file, asset_options)
    return shadow_asset

def load_colored_cube(gym, sim, isFixed):
    asset_file = "urdf/objects/cube_multicolor.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = isFixed
    asset_options.flip_visual_attachments = True
    asset_options.use_mesh_materials = True
    colored_cube_asset = gym.load_asset(sim, "../assets", asset_file, asset_options)

    return colored_cube_asset

def add_obstacles(sim, gym, env, environment_type, index):
    if environment_type == "arena":
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

def add_store(sim, gym, env, table_asset, shelf_asset, product_asset, index):

    # add table and shelf
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", index, 0)
    shelf_handle = gym.create_actor(env, shelf_asset, shelf_pose, "shelf", index, 0)
    hageslag_l = 0.062
    hageslag_h = 0.103
    box_handle = add_box(sim, gym, env, hageslag_h, hageslag_l, hageslag_l, box_pose, color_vec_crate, False, "product", index)
    box_props = gym.get_actor_rigid_body_properties(env, box_handle)
    box_props[0].mass = 0.45 + (0.7*(np.random.rand()-0.5))
    gym.set_actor_rigid_body_properties(env, box_handle, box_props)  
    box_props = gym.get_actor_rigid_body_properties(env, box_handle)
    print(box_props[0].mass)
    # mug_handle = gym.create_actor(env, mug_asset, box_pose, "mug", i, 0)
    #product_handle = gym.create_actor(env, product_asset, product_pose, "product", index, 0)
    return table_handle

def get_default_franka_state(gym, robot_asset):

    franka_dof_props = gym.get_asset_dof_properties(robot_asset)
    franka_lower_limits = franka_dof_props["lower"]
    franka_upper_limits = franka_dof_props["upper"]
    franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
    
    # default dof states and position targets
    franka_num_dofs = gym.get_asset_dof_count(robot_asset)
    default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
    default_dof_pos[:10] = franka_mids[:10]
    # grippers open
    # default_dof_pos[7:] = franka_lower_limits[7:]
    default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos
    default_dof_state["pos"][3] = -2
    print(default_dof_state["pos"])

    return default_dof_state

def create_robot_arena(gym, sim, num_envs, spacing, robot_asset, pose, viewer, environment_type, control_type = "vel_control"):
    # Some common handles for later use
    envs = []
    robot_handles = []
    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 6, 8), gymapi.Vec3(1.5, 0, 0))
    
    # Environment specific parameters and assets
    if environment_type == "arena" or environment_type == "battery":
        wall_size = 8
        wall_thickness = 0.1
    elif environment_type == "lab":
        wall_size = 5
        wall_thickness = 0.05
    elif environment_type == "store":
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1, 1, 1.5), gymapi.Vec3(-0.2, -0.2, 0.2))
        mug_asset = load_mug(gym, sim)
        hageslag_asset = load_hageslag(gym, sim)
        shelf_asset = load_shelf(gym, sim)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
    elif environment_type == "shadow":
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1, -1, 1.5), gymapi.Vec3(0., -0.4, 1))
        cube_asset = load_colored_cube(gym, sim, False)
        cube_pose = gymapi.Transform()
        cube_pose.p = gymapi.Vec3(-0.01, -0.42, 1.)

        cube_target_asset = load_colored_cube(gym, sim, True)
        cube_target_pose = gymapi.Transform()
        cube_target_pose.p = gymapi.Vec3(-0.01, -0.42, 1.3)
        cube_target_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 1, 1), np.random.uniform(-math.pi, math.pi))
    elif environment_type == "storm":
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 0, 0.6), gymapi.Vec3(-1, 0, 0))

    for i in range(num_envs):
        # Create env
        env = gym.create_env(sim, gymapi.Vec3(-spacing, 0.0, -spacing), gymapi.Vec3(spacing, spacing, spacing), num_per_row)
        envs.append(env)
        
        if environment_type == "store":
            table_handle = add_store(sim, gym, env, table_asset, shelf_asset, hageslag_asset, i)
            robot_handle = gym.create_actor(env, robot_asset, franka_pose, "franka", i, 2)
            # configure franka dofs
            if 'default_dof_state' not in locals():
                default_dof_state = get_default_franka_state(gym, robot_asset)
            gym.set_actor_dof_states(env, robot_handle, default_dof_state, gymapi.STATE_ALL)

            # Table friction randomized   
            shape_props = gym.get_actor_rigid_shape_properties(env, table_handle)
            shape_props[0].friction = np.random.uniform(0.1, 1)
            shape_props[0].torsion_friction = np.random.uniform(0.001, 0.01)
            gym.set_actor_rigid_shape_properties(env, table_handle, shape_props)
            # print(shape_props[0].friction)

        elif environment_type == "arena" or environment_type == "battery" or environment_type == "lab":
            add_arena(sim, gym, env, wall_size, wall_thickness, 0, 0, i) # Wall size, wall thickness, origin_x, origin_y, index
            # Add obstacles
            add_obstacles(sim, gym, env, environment_type, index = i)
            # Add robot
            robot_handle = gym.create_actor(env, robot_asset, pose, "robot", i, 0)
        elif environment_type == "shadow":
            # Add cube 
            cube_handle = gym.create_actor(env, cube_asset, cube_pose, "cube", i, 0)
            # Add target reference
            cube_target_handle = gym.create_actor(env, cube_target_asset, cube_target_pose, "cube", i, 0)
            # Add hand
            robot_handle = gym.create_actor(env, robot_asset, pose, "shadow_hand", i, 0)
            # shadow_num_dofs = gym.get_asset_dof_count(robot_asset)
            # default_dof_pos = np.zeros(shadow_num_dofs, dtype=np.float32)
            # default_dof_pos[1] = -0.5   # Palm
            # default_dof_pos[2] = 0.2   # Palm
            # default_dof_pos[3] = 0.4   # Index
            # default_dof_pos[4] = 0.6   # Index
            # default_dof_pos[5] = 0.6   # Index

            # # default_dof_pos[3] = 0
            # default_dof_state = np.zeros(shadow_num_dofs, gymapi.DofState.dtype)
            # default_dof_state["pos"] = default_dof_pos
            # gym.set_actor_dof_states(env, robot_handle, default_dof_state, gymapi.STATE_ALL)
        elif environment_type == "storm":
            # Add boxes
            #add_box(sim, gym, env, width, height, depth, pose, color, isFixed, name, index):

            add_box(sim, gym, env, 0.3, 0.1, 0.4, box1_pose_storm, color_vec_box1, False, 'box1', i)
            add_box(sim, gym, env, 0.3, 0.1, 0.5, box2_pose_storm, color_vec_box1, False, 'box2', i)

            # Add sphere
            # add sphere
            pose = gymapi.Transform()
            pose.p, pose.r = gymapi.Vec3(0.4, 0.4, 0.1), gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            gym.create_actor(env, gym.create_sphere(sim, 0.1, None), pose, "sphere", i, 0)

            # add goal
            pose = gymapi.Transform()
            pose.p, pose.r = gymapi.Vec3(0.5, .4, 0.2), gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            asset_options_objects = gymapi.AssetOptions()
            asset_options_objects.fix_base_link = True
            goal_handle = gym.create_actor(env, gym.create_sphere(sim, 0.01, asset_options_objects), pose, "goal", -2, 0) # No collision with goal
            gym.set_rigid_body_color(env, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color_vec_crate)

            # Add Panda
            robot_handle = gym.create_actor(env, robot_asset, franka_pose, "franka", i, 0)
            # configure franka dofs
            if 'default_dof_state' not in locals():
                default_dof_state = get_default_franka_state(gym, robot_asset)
            gym.set_actor_dof_states(env, robot_handle, default_dof_state, gymapi.STATE_ALL)

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
            props["damping"].fill(1000.0)
        else:
            print("Invalid control type!")

        props["friction"].fill(10.0)
        gym.set_actor_dof_properties(env, robot_handle, props)
        
        # panda_stick_rigid_body_names = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'panda_ee_finger', 'panda_ee_tip']
        # if gym.get_asset_rigid_body_names(robot_asset) == panda_stick_rigid_body_names:
        #     shape_props = gym.get_actor_rigid_shape_properties(env, robot_handle)
        #     for index in range(len(shape_props)-2):
        #         shape_props[index].friction = 1000000000.
        #         shape_props[index].torsion_friction = 100000000.
        #         shape_props[index].rolling_friction = 100000000.
        #     gym.set_actor_rigid_shape_properties(env, robot_handle, shape_props)

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

    print('Ready to start')
    return envs, robot_handles
