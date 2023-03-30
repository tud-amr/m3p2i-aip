from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
import numpy as np
import utils.env_conf as env_conf
import numpy as np
import utils.path_utils as path_utils
import os

# Parse arguments
args = path_utils.load_yaml(os.path.join(path_utils.get_params_path(),'physx.yml')) # dictionary

# Configure sim
def configure_sim(dt=0.05):
    # Get default set of parameters
    sim_params = gymapi.SimParams()
    # Set common parameters
    sim_params.dt = dt
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    # Set PhysX-specific parameters
    sim_params.use_gpu_pipeline = args['use_gpu']
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args['num_threads']
    sim_params.physx.use_gpu = args['use_gpu']
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    return sim_params

# Creating gym
def config_gym(viewer, dt):
    params = configure_sim(dt)
    gym = gymapi.acquire_gym()
    physics_engine = args['physics_engine']
    if(physics_engine=='physx'):
        physics_engine = gymapi.SIM_PHYSX
    elif(physics_engine == 'flex'):
        physics_engine = gymapi.SIM_FLEX
    sim = gym.create_sim(args['compute_device_id'], args['graphics_device_id'], physics_engine, params)
    if viewer:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        # Subscribe to input events.
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "left")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "down")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "right")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "up")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_1, "1")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_2, "2")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_3, "3")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_4, "4")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_5, "5")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_6, "6")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_7, "7")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_8, "8")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_9, "9")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
    else:
        viewer = None
    # Add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0
    gym.add_ground(sim, plane_params)
    return gym, sim, viewer

# Make the environment and simulation
def make(allow_viewer, num_envs, spacing, robot, obstacle_type, control_type = "vel_control", set_light = False, dt=0.05):
    # Configure gym
    gym, sim, viewer = config_gym(allow_viewer, dt)
    # Set robot initial pose
    robot_init_pose = gymapi.Transform()
    robot_init_pose.p = gymapi.Vec3(0.0, 0.0, 0.05)
    # Load robot
    robot_asset = env_conf.load_robot(robot, gym, sim)
    # Create the arena(s) with robots
    envs, robot_handles = env_conf.create_robot_arena(gym, sim, num_envs, spacing, robot_asset, robot_init_pose, viewer, obstacle_type, control_type)
    # Prepare
    gym.prepare_sim(sim)
    # Set light rendering
    if set_light:
        light_index = 3
        intensity = gymapi.Vec3(0.8, 0.8, 0.8)
        ambient = gymapi.Vec3(0.1, 0.1, 0.1)
        direction = gymapi.Vec3(1.5, 6.0, 8.0)
        gym.set_light_parameters(sim, light_index, intensity, ambient, direction)
    return gym, sim, viewer, envs, robot_handles

# Acquire states information
def acquire_states(gym, sim, params, flag="none"):
    if flag == "sim":
        num_envs = params.sim_num_envs
    else:
        num_envs = params.num_envs
    # Get dof state tensor
    _dof_states = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dof_states)
    num_dofs = gym.get_sim_dof_count(sim)
    num_actors = gym.get_sim_actor_count(sim)
    num_bodies = gym.get_sim_rigid_body_count(sim)
    dofs_per_robot = int(num_dofs/num_envs) # dofs_per_robot * 2 = nx
    actors_per_env = int(num_actors/num_envs)
    bodies_per_env = int(num_bodies/num_envs)

    # Acquire root state tensor
    _root_states = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(_root_states)
    shaped_root_states = root_states.reshape([num_envs, actors_per_env, 13])

    # Acquire rigid body states
    _rb_states = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(_rb_states)

    # Refresh the states
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)

    # Get 2D pos of block and robot
    if params.block_index != "None":
        block_pos = shaped_root_states[:, params.block_index, :2] # [num_envs, 2]
    else:
        block_pos = "None"
    if params.robot in ["boxer", "albert", "husky"]:
        robot_pos = shaped_root_states[:, -1, :2]   # [num_envs, 2]
        robot_vel = shaped_root_states[:, -1, 7:9]  # [num_envs, 2]
        robot_states = shaped_root_states[:, -1, :] # [num_envs, 13]
    elif params.robot in ["point_robot", "heijn"]:
        robot_pos = dof_states[:, 0].reshape([num_envs, dofs_per_robot])[:, :2] # [num_envs, 2]
        robot_vel = dof_states[:, 1].reshape([num_envs, dofs_per_robot])[:, :2] # [num_envs, 2]
        robot_states = dof_states.reshape([num_envs, dofs_per_robot*2]) # [num_envs, 4] or [num_envs, 6] for each row [pos1, vel1, pos2, vel2...]

    states_dict = {"dof_states": dof_states,
                   "root_states": root_states,
                   "shaped_root_states": shaped_root_states,
                   "rb_states": rb_states,
                   "robot_states": robot_states,
                   "num_dofs": num_dofs,
                   "num_actors": num_actors,
                   "dofs_per_robot": dofs_per_robot,
                   "actors_per_env": actors_per_env, 
                   "bodies_per_env": bodies_per_env, 
                   "robot_pos": robot_pos,
                   "block_pos": block_pos,
                   "robot_vel": robot_vel}

    # Print relevant info
    if params.print_flag:
        print("root_states", root_states.size())
        print('number of DOFs:', num_dofs) # num_envs * dof_per_actor
        print("dof_states size:", dof_states.size()) # [num_dofs, 2]
        print("pos", dof_states[:,0])
        print("vel", dof_states[:,1])
        print("actor num", num_actors)
        print("bodies num", num_bodies)

    return states_dict

# Visualize optimal trajectory
def visualize_traj(gym, viewer, env, actions, dof_states):
    # Get states and velocity
    # Note: the coordinate of the states is different from the visualization
    vels = actions.cpu().clone().numpy()
    dof_states_np = -dof_states.cpu().clone().numpy()
    curr_pos = dof_states_np[:, 0]
    n_steps = vels.shape[0]

    # Initialize array
    pos_array = np.zeros((n_steps+1, 3), dtype=np.float32)
    line_array = np.zeros((n_steps, 6), dtype=np.float32)
    pos_array[0, :] = [-curr_pos[1], -curr_pos[0], 0.15]
    color_array = np.zeros((n_steps, 3), dtype=np.float32)
    color_array[:, 2] = 255     # blue
    dt = 1.0 / 50.0
    for i in range(1, n_steps+1):
        pos_array[i, :] = [pos_array[i-1, 0] + dt*vels[i-1, 1], pos_array[i-1, 1] + dt*vels[i-1, 0], 0.2]
        line_array[i-1, :] = np.concatenate((pos_array[i-1, :], pos_array[i, :]))

    # Draw lines
    gym.add_lines(viewer, env, n_steps, line_array, color_array)

# Visualize rollouts trajectory
def visualize_rollouts(gym, viewer, env, states):
    # Initialize array
    n_steps = states.shape[0] - 1
    line_array = np.zeros((n_steps, 6), dtype=np.float32)
    color_array = np.zeros((n_steps, 3), dtype=np.float32)
    color_array[:, 1] = 255     # green
    for i in range(n_steps):
        pos = [states[i, 1], states[i, 0], 0.1, states[i+1, 1], states[i+1, 0], 0.1]
        line_array[i, :] = pos

    # Draw lines
    gym.add_lines(viewer, env, n_steps, line_array, color_array)

# Step the simulation
def step(gym, sim):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

# Refresh the states
def refresh_states(gym, sim):
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)

# Gym rendering 
def step_rendering(gym, sim, viewer, sync_frame_time=False):
    if viewer is not None:
        # Step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        if sync_frame_time:
            gym.sync_frame_time(sim)

# Time logging
def time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs, freq_data = np.zeros(2)):
    t = gym.get_elapsed_time(sim)
    if t >= next_fps_report:
        t2 = gym.get_elapsed_time(sim)
        fps = frame_count / (t2 - t1)
        print("FPS %.1f (%.1f)" % (fps, fps * num_envs))
        frame_count = 0
        t1 = gym.get_elapsed_time(sim)
        next_fps_report = t1 + 1
        if freq_data.any() != 0:
            print('Task freq', freq_data[0])
            print('Motion freq', freq_data[1])
    frame_count += 1
    return next_fps_report, frame_count, t1

# Destroy the simulation
def destroy_sim(gym, sim, viewer):
    print("Done")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

# Control using keyboard
def keyboard_control(gym, sim, viewer, robot, num_dofs, num_envs, dof_states, control_type = "vel_control"):
    # Set targets for different robots
    if robot == "point_robot":
        zero_vel = torch.zeros(num_dofs, dtype=torch.float32, device="cuda:0")
        up_vel = torch.tensor([0, -2], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        down_vel = torch.tensor([0, 2], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        left_vel = torch.tensor([2, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        right_vel = torch.tensor([-2, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        vel_targets = {"up":up_vel, "down":down_vel, "left":left_vel, "right":right_vel}
    elif robot == "boxer":
        max_vel = 5
        zero_vel = torch.zeros(1, num_dofs, dtype=torch.float32, device="cuda:0")
        left_vel = torch.tensor([-max_vel, max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        down_vel = torch.tensor([-max_vel, -max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        up_vel = torch.tensor([max_vel, max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        right_vel = torch.tensor([max_vel, -max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        vel_targets = {"up":up_vel, "down":down_vel, "left":left_vel, "right":right_vel}
    elif robot == "albert":
        max_vel = 5
        zero_vel = torch.zeros(num_dofs, dtype=torch.float32, device="cuda:0")
        joint_1 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        joint_2 = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        joint_3 = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        joint_4 = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        joint_5 = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        joint_6 = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        joint_7 = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        joint_8 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        joint_9 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        left_vel = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, -max_vel, max_vel, -max_vel, max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        down_vel = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, -max_vel, -max_vel, -max_vel, -max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        up_vel = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, max_vel, max_vel, max_vel, max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        right_vel = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, max_vel, -max_vel, max_vel, -max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        vel_targets = {"up":up_vel, "down":down_vel, "left":left_vel, "right":right_vel, 
                        "1":joint_1, "2":joint_2, "3":joint_3, "4":joint_4, "5":joint_5,
                        "6":joint_6, "7":joint_7, "8":joint_8, "9":joint_9}
    elif robot == "husky":
        max_vel = 5
        zero_vel = torch.zeros(1, num_dofs, dtype=torch.float32, device="cuda:0")
        left_vel = torch.tensor([-max_vel, max_vel, -max_vel, max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        down_vel = torch.tensor([-max_vel, -max_vel, -max_vel, -max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        up_vel = torch.tensor([max_vel, max_vel, max_vel, max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        right_vel = torch.tensor([max_vel, -max_vel, max_vel, -max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        vel_targets = {"up":up_vel, "down":down_vel, "left":left_vel, "right":right_vel}
    elif robot == "heijn":
        max_vel = 5
        zero_vel = torch.zeros(1, num_dofs, dtype=torch.float32, device="cuda:0")
        x_vel = torch.tensor([max_vel, 0, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        y_vel = torch.tensor([0, max_vel, 0], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        theta_vel = torch.tensor([0, 0, max_vel], dtype=torch.float32, device="cuda:0").repeat(num_envs)
        vel_targets = {"up":-y_vel, "down":y_vel, "left":x_vel, "right":-x_vel, "1":theta_vel, "2":-theta_vel }

    # Respond the keyboard
    for evt in gym.query_viewer_action_events(viewer):
        if evt.value > 0:
            if control_type == "pos_control":
                current_pos = dof_states[:,0]
                gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(current_pos+vel_targets[evt.action]))
            if control_type == "vel_control":
                gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(vel_targets[evt.action]))
            if control_type == "force_control":
                gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(vel_targets[evt.action]))
        else:
            if control_type == "pos_control":
                pass
            if control_type == "vel_control":
                gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(zero_vel))
            if control_type == "force_control":
                gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(zero_vel))

# Update movement of dynamic obstacle
def update_dyn_obs(gym, sim, num_actors, num_envs, count):
    gym.refresh_actor_root_state_tensor(sim)
    _root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_tensor = gymtorch.wrap_tensor(_root_tensor)
    root_positions = root_tensor[:, 0:3] # [56, 3]
    root_linvels = root_tensor[:, 7:10]

    offsets = torch.tensor([0.02, 0.02, 0], dtype=torch.float32, device="cuda:0").repeat(num_actors, 1)
    size = 200
    if count % size > size/4 and count % size < size/4*3:
        root_positions += offsets
    else:
        root_positions -= offsets
    dyn_obs_index = 5
    index_offset = num_actors - dyn_obs_index
    indice_list = []
    for i in range(num_envs):
        indice_list.append((i+1)*num_actors/num_envs-index_offset)
    actor_indices = torch.tensor(indice_list, dtype=torch.int32, device="cuda:0")
    gym.set_actor_root_state_tensor_indexed(sim, _root_tensor, gymtorch.unwrap_tensor(actor_indices), num_envs)