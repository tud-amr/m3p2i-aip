from isaacgym import gymapi
from isaacgym import gymtorch
from dataclasses import dataclass
import  m3p2i_aip.utils.isaacgym_utils.actor_utils as actor_utils
import numpy as np
import torch
@dataclass
class IsaacGymConfig(object):
    dt: float = 0.01 # 0.01
    substeps: int = 2
    use_gpu_pipeline: bool = True
    num_threads: int = 8
    viewer: bool = False
    spacing: float = 2.0 # !!
    # panda_camera_pos1: List[float] = [0, 1.5, 2.8]
    # panda_camera_pos2: List[float] = [0, 0, 1]
    # point_camera_pos1: List[float] = [1.5, 6, 8]
    # point_camera_pos2: List[float] = [1.5, 0, 0]


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
        env_type: str = "point_env",
        # actors: List[str],
        # init_positions: List[List[float]] = None,
        num_envs: int = 1,
        viewer: bool = False,
        device: str = "cuda:0",
        # interactive_goal = True
    ):
        self._gym = gymapi.acquire_gym()
        self.env_type = env_type
        self.env_cfg = actor_utils.load_env_cfgs(env_type)
        self.device = device

        self.cfg = cfg
        if viewer:
            self.cfg.viewer = viewer
        # self.interactive_goal = interactive_goal
        self.num_envs = num_envs
        # self.camera_pos1 = {"point_env":self.cfg.point_camera_pos1, "panda_env":self.cfg.panda_camera_pos1}
        # self.camera_pos2 = {"point_env":self.cfg.point_camera_pos2, "panda_env":self.cfg.panda_camera_pos2}
        # self.restarted = 1
        self.start_sim()

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

        self.creat_env()

        self._gym.prepare_sim(self._sim)

        self.set_initial_joint_pose()

    def set_initial_joint_pose(self):
        # set initial joint poses
        robots = [a for a in self.env_cfg if a.type == "robot"]
        for robot in robots:
            dof_state = []
            if robot.init_joint_pose:
                dof_state += robot.init_joint_pose
                print(dof_state)
            else:
                dof_state += (
                    [0] * 2 * self._gym.get_actor_dof_count(self.envs[0], robot.handle)
                )
        dof_state = (
            torch.tensor(dof_state, device=self.device)
            .type(torch.float32)
            .repeat(self.num_envs, 1)
        )
        self._gym.set_dof_state_tensor(self._sim, gymtorch.unwrap_tensor(dof_state))
        self._gym.refresh_dof_state_tensor(self._sim)

    def creat_env(self):
        # Load / create assets for all actors in the envs
        env_actor_assets = []
        for actor_cfg in self.env_cfg:
            asset = actor_utils.load_asset(self._gym, self._sim, actor_cfg)
            env_actor_assets.append(asset)

        # self._gym.viewer_camera_look_at(self.viewer, None, 
        #                                 gymapi.Vec3(*self.camera_pos1[self.env_type]), 
        #                                 gymapi.Vec3(*self.camera_pos2[self.env_type]))

        if self.env_type == 'panda_env':
            self._gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(0, 1.5, 2.8), gymapi.Vec3(0, 0, 1))
        else:
            self._gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(1.5, 6, 8), gymapi.Vec3(1.5, 0, 0))

        # Create envs and fill with assets
        self.envs = []
        for env_idx in range(self.num_envs):
            env = self._gym.create_env(
                self._sim,
                gymapi.Vec3(-self.cfg.spacing, 0.0, -self.cfg.spacing),
                gymapi.Vec3(self.cfg.spacing, self.cfg.spacing, self.cfg.spacing),
                int(self.num_envs**0.5),
            )

            for actor_asset, actor_cfg in zip(env_actor_assets, self.env_cfg):
                actor_cfg.handle = self._create_actor(
                    env, env_idx, actor_asset, actor_cfg
                )
            self.envs.append(env)

    def _create_actor(self, env, env_idx, asset, actor: actor_utils.ActorWrapper) -> int:
        if actor.noise_sigma_size is not None:
            asset = actor_utils.load_asset(self._gym, self._sim, actor)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*actor.init_pos)
        pose.r = gymapi.Quat(*actor.init_ori)
        handle = self._gym.create_actor(
            env=env,
            asset=asset,
            pose=pose,
            name=actor.name,
            group=env_idx if actor.collision else env_idx + self.num_envs, #
        )

        if actor.noise_sigma_size:
            actor.color = np.random.rand(3)

        self._gym.set_rigid_body_color(
            env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*actor.color)
        )
        props = self._gym.get_actor_rigid_body_properties(env, handle)
        # actor_mass_noise = np.random.uniform(
        #     -actor.noise_percentage_mass * actor.mass,
        #     actor.noise_percentage_mass * actor.mass,
        # )
        # props[0].mass = actor.mass + actor_mass_noise
        # self._gym.set_actor_rigid_body_properties(env, handle, props)
        print("mass", props[0].mass)

        body_names = self._gym.get_actor_rigid_body_names(env, handle)
        body_to_shape = self._gym.get_actor_rigid_body_shape_indices(env, handle)
        caster_shapes = [
            b.start
            for body_idx, b in enumerate(body_to_shape)
            if actor.caster_links is not None
            and body_names[body_idx] in actor.caster_links
        ]

        props = self._gym.get_actor_rigid_shape_properties(env, handle)
        for i, p in enumerate(props):
            actor_friction_noise = np.random.uniform(
                -actor.noise_percentage_friction * actor.friction,
                actor.noise_percentage_friction * actor.friction,
            )
            p.friction = actor.friction + actor_friction_noise #
            p.torsion_friction = np.random.uniform(0.001, 0.01)
            p.rolling_friction = actor.friction + actor_friction_noise

            if i in caster_shapes:
                p.friction = 0
                p.torsion_friction = 0
                p.rolling_friction = 0

        self._gym.set_actor_rigid_shape_properties(env, handle, props)

        if actor.type == "robot":
            # TODO: Currently the robot_rigid_body_viz_idx is only supported for a single robot case.
            if actor.visualize_link:
                self.robot_rigid_body_viz_idx = self._gym.find_actor_rigid_body_index(
                    env, handle, actor.visualize_link, gymapi.IndexDomain.DOMAIN_ENV
                )

            props = self._gym.get_asset_dof_properties(asset)
            if actor.name == "panda": #!
                props["driveMode"][7:].fill(gymapi.DOF_MODE_VEL)
                props["stiffness"][7:].fill(800.0)
                props["damping"][7:].fill(40.0)
            elif actor.dof_mode == "effort":
                props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
                props["stiffness"].fill(0.0)
                props["armature"].fill(0.0)
                props["damping"].fill(10.0)
            elif actor.dof_mode == "velocity":
                props["driveMode"].fill(gymapi.DOF_MODE_VEL)
                props["stiffness"].fill(0.0)
                props["damping"].fill(600.0)
            elif actor.dof_mode == "position":
                props["driveMode"].fill(gymapi.DOF_MODE_POS)
                props["stiffness"].fill(80.0)
                props["damping"].fill(0.0)
            else:
                raise ValueError("Invalid dof_mode")
            self._gym.set_actor_dof_properties(env, handle, props)
        return handle
    
    def step(self):
        self._gym.simulate(self._sim)
        self._gym.fetch_results(self._sim, True)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)

        if self.viewer is not None:
            self._gym.step_graphics(self._sim)
            self._gym.draw_viewer(self.viewer, self._sim, False)

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