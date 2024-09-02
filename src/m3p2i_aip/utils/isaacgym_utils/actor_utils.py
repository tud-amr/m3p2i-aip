
from isaacgym import gymapi
from enum import Enum
from yaml import SafeLoader
import yaml, pathlib, numpy as np 
from typing import List, Optional, Any
from dataclasses import dataclass, field
import m3p2i_aip.utils.path_utils as path_utils

class SupportedActorTypes(Enum):
    Axis = 1
    Robot = 2
    Sphere = 3
    Box = 4

@dataclass
class ActorWrapper:
    type: SupportedActorTypes
    name: str
    dof_mode: str = "velocity"
    init_pos: List[float] = field(default_factory=lambda: [0, 0, 0])
    init_pos_on_table: List[float] = field(default_factory=lambda: [0, 0, 0])
    init_pos_on_shelf: List[float] = field(default_factory=lambda: [0, 0, 0])
    init_ori: List[float] = field(default_factory=lambda: [0, 0, 0, 1])
    size: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    mass: float = 1.0  # kg
    color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    fixed: bool = False
    collision: bool = True
    friction: float = 1.0 # 1.0
    handle: Optional[int] = None
    flip_visual: bool = False
    urdf_file: str = None
    visualize_link: str = None
    gravity: bool = True
    differential_drive: bool = False
    init_joint_pose: List[float] = None
    wheel_radius: Optional[float] = None
    wheel_base: Optional[float] = None
    wheel_count: Optional[float] = None
    left_wheel_joints: Optional[List[str]] = None
    right_wheel_joints: Optional[List[str]] = None
    caster_links: Optional[List[str]] = None
    noise_sigma_size: Optional[List[float]] = None
    noise_percentage_mass: float = 0.0
    noise_percentage_friction: float = 0.0

def load_asset(gym, sim, actor_cfg):
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = actor_cfg.fixed
    asset_options.disable_gravity = not actor_cfg.gravity
    asset_root_path = path_utils.get_assets_path()

    if actor_cfg.type == "robot":
        asset_file = "urdf/" + actor_cfg.urdf_file
        asset_options.flip_visual_attachments = actor_cfg.flip_visual
        actor_asset = gym.load_asset(
            sim=sim,
            rootpath=asset_root_path,
            filename=asset_file,
            options=asset_options,
        )
    elif actor_cfg.type == "box":
        if actor_cfg.noise_sigma_size is not None:
            noise_sigma = np.array(actor_cfg.noise_sigma_size)
        else:
            noise_sigma = np.zeros((3,))
        noise = np.random.normal(loc=0, scale=noise_sigma, size=3)
        actor_asset = gym.create_box(
            sim=sim,
            width=actor_cfg.size[0] + noise[0],
            height=actor_cfg.size[1] + noise[1],
            depth=actor_cfg.size[2] + noise[2],
            options=asset_options,
        )
    elif actor_cfg.type == "sphere":
        if actor_cfg.noise_sigma_size is not None:
            noise_sigma = np.array(actor_cfg.noise_sigma_size)
        else:
            noise_sigma = np.zeros((1,))
        noise = np.random.normal(loc=0, scale=noise_sigma, size=1)
        actor_asset = gym.create_sphere(
            sim=sim,
            radius=actor_cfg.size[0] + noise[0],
            options=asset_options,
        )
    else:
        raise NotImplementedError(
            f"actor asset of type {actor_cfg.type} is not yet implemented!"
        )

    return actor_asset

def load_env_cfgs(env_type: str) -> List[ActorWrapper]:
    actor_cfgs = []
    env_path = path_utils.get_config_path() + env_type
    for file in pathlib.Path(env_path).iterdir():
        with open (f"{file}") as f:
            actor_cfgs.append(ActorWrapper(**yaml.load(f, Loader=SafeLoader)))

    return actor_cfgs

if __name__== "__main__":
    actor_cfgs = load_env_cfgs("point_env")
    print(actor_cfgs)