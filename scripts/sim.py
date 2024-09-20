from isaacgym import gymtorch
import torch, hydra, zerorpc, time
from m3p2i_aip.config.config_store import ExampleConfig
import  m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes
from m3p2i_aip.utils.skill_utils import check_and_apply_suction, time_tracking
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

'''
Run in the command line:
    python3 sim.py
    python3 sim.py task=pull
    python3 sim.py task=push_pull
    python3 sim.py -cn config_panda
    python3 sim.py -cn config_panda multi_modal=True cube_on_shelf=True
'''

@hydra.main(version_base=None, config_path="../src/m3p2i_aip/config", config_name="config_point")
def run_sim(cfg: ExampleConfig):
    sim = wrapper.IsaacGymWrapper(
        cfg.isaacgym,
        cfg.env_type,
        num_envs=1,
        viewer=True,
        device=cfg.mppi.device,
        cube_on_shelf=cfg.cube_on_shelf,
    )

    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    print("Server found and wait for the viewer")
    for _ in range(150):
        sim.step()
    print("Start simulation!")

    t = time.time()
    for i in range(10000):
        sim.update_dyn_obs(i)
        sim.play_with_cube()

        action = bytes_to_torch(
            planner.run_tamp(
                torch_to_bytes(sim._dof_state), torch_to_bytes(sim._root_state))
        )
        sim.set_dof_velocity_target_tensor(action)

        cfg.suction_active = bytes_to_torch(
            planner.get_suction()
        )
        check_and_apply_suction(cfg, sim, action)

        sim.step()

        # sim.visualize_trajs(
        #     bytes_to_torch(planner.get_trajs())
        # )
        
        t = time_tracking(t, cfg)

if __name__== "__main__":
    run_sim()