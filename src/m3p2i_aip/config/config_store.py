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
    kp_suction: int
    suction_active: bool
    multi_modal: bool
    cube_on_shelf: bool

cs = ConfigStore.instance()
cs.store(name="config_point", node=ExampleConfig)
cs.store(name="config_panda", node=ExampleConfig)
cs.store(group="mppi", name="base_mppi", node=MPPIConfig)
cs.store(group="isaacgym", name="base_isaacgym", node=IsaacGymConfig)