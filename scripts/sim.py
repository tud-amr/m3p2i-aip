from isaacgym import gymtorch
import torch, hydra, zerorpc
from m3p2i_aip.config.config_store import ExampleConfig
import  m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes
from m3p2i_aip.utils.skill_utils import check_suction_condition, calculate_suction
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

'''
Run in the command line:
    python3 sim.py
    python3 sim.py task=pull
'''

@hydra.main(version_base=None, config_path="../src/m3p2i_aip/config", config_name="config_point")
def run_sim(cfg: ExampleConfig):

    sim = wrapper.IsaacGymWrapper(
        cfg.isaacgym,
        cfg.env_type,
        num_envs=1,
        viewer=True,
        device=cfg.mppi.device,
    )

    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    print("Server found and wait for the viewer")
    for _ in range(150):
        sim.step()
    print("Start simulation!")

    while True:
        # print("dof", sim._dof_state[0], "root", sim._root_state[0])
        action = bytes_to_torch(
            planner.run_tamp(
                torch_to_bytes(sim._dof_state), torch_to_bytes(sim._root_state))
        )
        # print("task", cfg.task, "action", action)
        sim.set_dof_velocity_target_tensor(action)

        if check_suction_condition(cfg, sim, action):
            suction_force = calculate_suction(cfg, sim)
            sim.apply_rigid_body_force_tensors(suction_force)

        # Step simulator
        sim.step()

if __name__== "__main__":
    run_sim()