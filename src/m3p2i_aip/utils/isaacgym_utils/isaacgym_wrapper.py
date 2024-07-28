from isaacgym import gymapi
from isaacgym import gymtorch
from dataclasses import dataclass

@dataclass
class IsaacGymConfig(object):
    dt: float = 0.05
    substeps: int = 2
    use_gpu_pipeline: bool = True
    num_threads: int = 8
    viewer: bool = False

def parse_isaacgym_config(cfg: IsaacGymConfig, device: str = "cuda:0") -> gymapi.SimParams:
    sim_params = gymapi.SimParams()
    sim_params.dt = cfg.dt
    sim_params.substeps = cfg.substeps
    sim_params.use_gpu_pipeline = device == "cuda:0"
    # sim_params.num_client_threads = cfg.num_client_threads
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.num_threads = cfg.num_threads
    # sim_params.physx.use_gpu = cfg.use_gpu_pipeline
    # sim_params.physx.friction_offset_threshold = 0.01
    # sim_params.physx.friction_correlation_distance = 0.001

    # return the configured params
    return sim_params

class IsaacGymWrapper:
    def __init__(
        self,
        cfg: IsaacGymConfig,
        # actors: List[str],
        # init_positions: List[List[float]] = None,
        num_envs: int = 1,
        viewer: bool = False,
        device: str = "cuda:0",
        # interactive_goal = True
    ):
        self._gym = gymapi.acquire_gym()
        # self.env_cfg = load_actor_cfgs(actors)
        self.device = device

        self.cfg = cfg
        if viewer:
            self.cfg.viewer = viewer
        # self.interactive_goal = interactive_goal
        self.num_envs = num_envs
        # self.restarted = 1
        self.start_sim()

    def initialize_keyboard_listeners(self):
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "left")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "down")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "right")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "up")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_1, "1")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_2, "2")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_3, "3")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_4, "4")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_5, "5")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_6, "6")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_7, "7")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_8, "8")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_9, "9")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "key_left")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "key_down")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "key_right")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "key_up")

    def add_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0
        self._gym.add_ground(self._sim, plane_params)

    def start_sim(self):
        self._sim = self._gym.create_sim(
            compute_device = 0,
            graphics_device = 0,
            type = gymapi.SIM_PHYSX,
            params = parse_isaacgym_config(self.cfg, self.device),
        )

        if self.cfg.viewer:
            self.viewer = self._gym.create_viewer(self._sim, gymapi.CameraProperties())
            self.initialize_keyboard_listeners()
        else:
            self.viewer = None

        self.add_ground_plane()