from isaacgym import gymtorch
from m3p2i_aip.planners.motion_planner import m3p2i
from m3p2i_aip.planners.task_planner import task_planner
from m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes
from m3p2i_aip.planners.motion_planner.cost_functions import Objective
import torch, time, copy, socket, numpy as np
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)
import  m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper
import hydra, zerorpc
from m3p2i_aip.config.config_store import ExampleConfig

'''
Run in the command line:
    python3 reactive_tamp.py task=navigation goal="[-1, -1]"
    python3 reactive_tamp.py task=push goal="[-1, -1]"
    python3 reactive_tamp.py task=pull goal="[0, 0]"
'''

class REACTIVE_TAMP:
    def __init__(self, cfg) -> None:

        self.sim = wrapper.IsaacGymWrapper(
            cfg.isaacgym,
            cfg.env_type,
            num_envs=cfg.mppi.num_samples,
            viewer=False,
            device=cfg.mppi.device,
        )
        self.cfg = cfg
        self.objective = Objective(cfg)

        # Choose the task planner
        if self.sim.env_type == "point_env":
            self.task_planner = task_planner.PLANNER_SIMPLE(cfg.task, cfg.goal)
        else:
            self.task_planner = task_planner.PLANNER_AIF_PANDA()
        self.task_success = False

        # Choose the motion planner
        self.motion_planner = m3p2i.M3P2I(cfg,
                                          dynamics = self.dynamics, 
                                          running_cost=self.running_cost)
        self.motion_planner.set_mode(mppi_mode = 'halton-spline', # 'halton-spline', 'simple'
                                     sample_method = 'halton',    # 'halton', 'random'
                                     multi_modal = cfg.multi_modal)
        self.prefer_pull = -1

    def run_tamp(self, dof_state, root_state):
        # Set rollout state from sim state
        self.sim._dof_state[:] = bytes_to_torch(dof_state)
        self.sim._root_state[:] = bytes_to_torch(root_state)
        self.sim._gym.set_dof_state_tensor(
            self.sim._sim, gymtorch.unwrap_tensor(self.sim._dof_state)
        )
        self.sim._gym.set_actor_root_state_tensor(
            self.sim._sim, gymtorch.unwrap_tensor(self.sim._root_state)
        )

        self.tamp_interface()
        
        if self.task_success:
            print("--------Task success--------")
            return torch_to_bytes(
                torch.zeros(self.sim.dofs_per_robot, device=self.cfg.mppi.device)
            )
        else:
            print("--------Compute optimal action--------")
            return torch_to_bytes(
                self.motion_planner.command(self.sim._dof_state[0])[0]
            )
    
    def dynamics(self, _, u, t=None):
        self.sim.set_dof_velocity_target_tensor(u)
        self.sim.step()
        states = torch.stack([self.sim.robot_pos[:, 0], 
                              self.sim.robot_vel[:, 0], 
                              self.sim.robot_pos[:, 1], 
                              self.sim.robot_vel[:, 1]], dim=1) # [num_envs, 4]
        return states, u
    
    def running_cost(self, _):
        return self.objective.compute_cost(self.sim)
    
    def tamp_interface(self):
        print("task", self.task_planner.task, "goal", self.task_planner.curr_goal)
        self.task_success = self.task_planner.check_task_success(self.sim)

@hydra.main(version_base=None, config_path="../src/m3p2i_aip/config", config_name="config_point")
def run_reactive_tamp(cfg: ExampleConfig):
    reactive_tamp = REACTIVE_TAMP(cfg)
    planner = zerorpc.Server(reactive_tamp)
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()

if __name__== "__main__":
    run_reactive_tamp()