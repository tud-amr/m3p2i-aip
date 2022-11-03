from isaacgym import gymapi
from isaacgym import gymutil

# parse arguments
args = gymutil.parse_arguments(description="Experiments")
args.use_gpu = True
# args.graphics_device_id = -1

# configure sim
def configure_sim():
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
    return sim_params

# creating gym
def config_gym(viewer):
    params = configure_sim()
    gym = gymapi.acquire_gym()
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, params)
    if viewer:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        # subscribe to input events. This allows input to be used to interact
        # with the simulation
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

# Step the simulation
def step(gym, sim):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

# Gym rendering 
def step_rendering(gym, sim, viewer):
    if viewer is not None:
        # Step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

# Time logging
def time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs):
    t = gym.get_elapsed_time(sim)
    if t >= next_fps_report:
        t2 = gym.get_elapsed_time(sim)
        fps = frame_count / (t2 - t1)
        print("FPS %.1f (%.1f)" % (fps, fps * num_envs))
        frame_count = 0
        t1 = gym.get_elapsed_time(sim)
        next_fps_report = t1 + 2.0
    frame_count += 1
    return next_fps_report, frame_count, t1

# Destroy the simulation
def destroy_sim(gym, sim, viewer):
    print("Done")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)