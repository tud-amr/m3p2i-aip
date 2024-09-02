from dataclasses import dataclass, field
from m3p2i_aip.planners.motion_planner.mppi import MPPIConfig
from m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper import IsaacGymConfig
from hydra.core.config_store import ConfigStore
from typing import List, Optional

@dataclass
class ExampleConfig:
    render: bool
    n_steps: int
    mppi: MPPIConfig
    isaacgym: IsaacGymConfig
    env_type: str
    task: str
    goal: List[float]
    nx: int
    actors: List[str]
    initial_actor_positions: List[List[float]]
    kp_suction: int = 0
    suction_active: bool = False
    multi_modal: bool = False
    pre_height_diff: float = 0.
    cube_on_shelf: bool = False

cs = ConfigStore.instance()
cs.store(name="config_point", node=ExampleConfig)
cs.store(name="config_panda", node=ExampleConfig)
cs.store(group="mppi", name="base_mppi", node=MPPIConfig)
cs.store(group="isaacgym", name="base_isaacgym", node=IsaacGymConfig)