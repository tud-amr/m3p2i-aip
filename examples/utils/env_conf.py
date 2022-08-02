from isaacgym import gymapi

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

movable_box_pose = gymapi.Transform()
movable_box_pose.p = gymapi.Vec3(0.5, 0.5, 0)

obstacle_pose = gymapi.Transform()
obstacle_pose.p = gymapi.Vec3(3, 3, 0)

goal_pose = gymapi.Transform()
goal_pose.p = gymapi.Vec3(-3, 3, 0)

docking_station_loc = [-3, -3]
recharge_pose = gymapi.Transform()
recharge_pose.p = gymapi.Vec3(docking_station_loc[1], docking_station_loc[0], 0)

color_vec_movable = gymapi.Vec3(0.5, 0.1, 0.7)
color_vec_fixed = gymapi.Vec3(0.8, 0.2, 0.2)
color_vec_walls= gymapi.Vec3(0.1, 0.1, 0.1)
color_vec_goal= gymapi.Vec3(0.5, 0.2, 0.7)
color_vec_recharge= gymapi.Vec3(0.0, 0.9, 0.3)

color_vec_battery_ok = gymapi.Vec3(0.0, 0.7, 0.5)
color_vec_battery_low = gymapi.Vec3(0.8, 0.5, 0.)
color_vec_battery_critical = gymapi.Vec3(0.8, 0.2, 0.2)