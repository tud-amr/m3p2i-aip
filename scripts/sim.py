from isaacgym import gymapi, gymtorch
import torch, time, numpy as np, socket
from m3p2i_aip.utils import data_transfer, skill_utils
from m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes
import hydra, zerorpc
import  m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from m3p2i_aip.config.config_store import ExampleConfig
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

'''
Run in the command line:
    python3 sim.py
'''

@hydra.main(version_base=None, config_path="../src/m3p2i_aip/config", config_name="config_point")
def run_sim(cfg: ExampleConfig):

    sim = wrapper.IsaacGymWrapper(
        cfg.isaacgym,
        cfg.env_type,
        num_envs=1,
        viewer=True,
    )

    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    print("Server found!")

    while True:
        # print("dof", sim._dof_state[0], "root", sim._root_state[0])
        action = bytes_to_torch(
            planner.set_rollout_sim(
                torch_to_bytes(sim._dof_state), torch_to_bytes(sim._root_state))
        )
        # print("task", cfg.task, "action", action)
        sim.set_dof_velocity_target_tensor(action[0])

        # Step simulator
        sim.step()

if __name__== "__main__":
    run_sim()