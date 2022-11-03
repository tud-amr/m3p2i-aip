from isaacgym import gymapi
from isaacgym import gymutil

# parse arguments
args = gymutil.parse_arguments(description="Experiments")
args.use_gpu = True
# args.graphics_device_id = -1

# configure sim
def configure_sim():
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 20.0
    sim_params.substeps = 2
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
