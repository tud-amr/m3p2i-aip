from isaacgym import gymtorch
import torch, hydra, zerorpc
from m3p2i_aip.planners.motion_planner import m3p2i
from m3p2i_aip.planners.task_planner import task_planner
from m3p2i_aip.config.config_store import ExampleConfig
import  m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from m3p2i_aip.planners.motion_planner.cost_functions import Objective
from m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

'''
Run in the command line:
    python3 reactive_tamp.py task=navigation goal="[-3, 3]"
    python3 reactive_tamp.py task=push goal="[-1, -1]"
    python3 reactive_tamp.py task=pull goal="[0, 0]"
    python3 reactive_tamp.py task=push_pull multi_modal=True goal="[-3.75, -3.75]"
    python3 reactive_tamp.py -cn config_panda
    python3 reactive_tamp.py -cn config_panda multi_modal=True cube_on_shelf=True
'''

class REACTIVE_TAMP:
    def __init__(self, cfg) -> None:

        self.sim = wrapper.IsaacGymWrapper(
            cfg.isaacgym,
            cfg.env_type,
            num_envs=cfg.mppi.num_samples,
            viewer=False,
            device=cfg.mppi.device,
            cube_on_shelf=cfg.cube_on_shelf,
        )
        self.cfg = cfg
        self.objective = Objective(cfg)

        self.task_planner = task_planner.set_task_planner(cfg)
        self.task_success = False
        
        self.motion_planner = m3p2i.M3P2I(
            cfg,
            dynamics=self.dynamics, 
            running_cost=self.running_cost
        )

    def run_tamp(self, dof_state, root_state):
        # Set rollout state from sim state
        self.sim._dof_state[:] = bytes_to_torch(dof_state)
        self.sim._root_state[:] = bytes_to_torch(root_state)
        self.sim.set_dof_state_tensor(self.sim._dof_state)
        self.sim.set_actor_root_state_tensor(self.sim._root_state)

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
        self.task_planner.update_plan(self.sim)
        print("task", self.task_planner.task, "goal", self.task_planner.curr_goal)
        self.motion_planner.update_gripper_command(self.task_planner.task)
        self.objective.update_objective(self.task_planner.task, self.task_planner.curr_goal)
        self.suction_active = self.motion_planner.get_pull_preference()
        self.task_success = self.task_planner.check_task_success(self.sim)

    def get_trajs(self):
        return torch_to_bytes(self.motion_planner.top_trajs)
    
    def get_suction(self):
        return torch_to_bytes(self.suction_active)

@hydra.main(version_base=None, config_path="../src/m3p2i_aip/config", config_name="config_point")
def run_reactive_tamp(cfg: ExampleConfig):
    reactive_tamp = REACTIVE_TAMP(cfg)
    planner = zerorpc.Server(reactive_tamp)
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()

if __name__== "__main__":
    run_reactive_tamp()