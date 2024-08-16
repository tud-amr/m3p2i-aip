from isaacgym import gymapi, gymtorch
import torch, time, numpy as np, socket
from m3p2i_aip.utils import data_transfer, skill_utils
from m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes
import hydra, zerorpc
import  m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from m3p2i_aip.config.config_store import ExampleConfig
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

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

        # print(planner.test_task_planner(122))
        # print(planner.motion_planner.K) # not working
        # print(planner.task_planner.curr_goal) # not working
        # print("dof", sim._dof_state[0], "root", sim._root_state[0])

        action = bytes_to_torch(
            planner.set_rollout_sim(
                torch_to_bytes(sim._dof_state[0]), torch_to_bytes(sim._root_state[0]))
        )
        
        # print("action", action)
        sim.set_dof_velocity_target_tensor(action[0])

        # Step simulator
        sim.step()


    # # Helper variables, same as in m3p2i
    # self.suction_active = params.suction_active
    # self.block_index = params.block_index
    # self.kp_suction = params.kp_suction

    # # Data logging
    # self.frame_count = 0
    # self.next_fps_report = 2.0
    # self.t1 = 0
    # self.count = 0
    # self.sim_time = np.array([])
    # self.dyn_obs_id = 5
    # self.dyn_obs_coll = 0
    # # Set server address
    # self.server_address = './uds_socket'

    # def run(self):
    #     with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    #         s.connect(self.server_address)
    #         t_prev = time.monotonic()

    #         while self.viewer is None or not self.gym.query_viewer_has_closed(self.viewer):
    #             # Reset the simulation when pressing 'R'
    #             reset_flag = False
    #             s.sendall(data_transfer.numpy_to_bytes(int(reset_flag)))
    #             self.task = s.recv(1024).decode('utf-8')

    #             # Send dof states to mppi and receive message
    #             s.sendall(data_transfer.torch_to_bytes(self.dof_states))
    #             self.curr_planner_task = s.recv(1024).decode('utf-8')

    #             # Send root states and receive optimal actions
    #             s.sendall(data_transfer.torch_to_bytes(self.root_states))
    #             b = s.recv(2**14)
    #             actions = data_transfer.bytes_to_torch(b)

    #             # Receive freq data
    #             s.sendall(b"freq data")
    #             b = s.recv(1024)
    #             freq_data = data_transfer.bytes_to_numpy(b)
    #             self.suction_active = int(freq_data[1])

    #             # Clear lines at the beginning
    #             self.gym.clear_lines(self.viewer)
                
    #             # Visualize top trajs
    #             if freq_data[0] != 0:
    #                 s.sendall(b"Visualize trajs")
    #                 _top_trajs = s.recv(2**12)
    #                 top_trajs = data_transfer.bytes_to_torch(_top_trajs)
    #                 sim_init.visualize_toptrajs(self.gym, self.viewer, self.envs[0], top_trajs, self.is_mobile_robot)
                
    #             # For multi-modal mppi
    #             if self.environment_type == 'normal':
    #                 dir_robot_bloc = (self.robot_pos-self.block_pos).squeeze(0)
    #                 check = torch.sum(actions[0] * dir_robot_bloc).item()
    #                 dis = torch.linalg.norm(dir_robot_bloc)
    #                 self.suction_active = False
    #                 if dis < 0.6 and check > 0 and self.curr_planner_task in ['pull', 'hybrid']:
    #                     self.suction_active = True

    #             # Apply forward kikematics and optimal action
    #             self.action = skill_utils.apply_fk(params.robot, actions[0])
    #             self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self.action))

    #             if self.suction_active:  
    #                 # Simulation of a magnetic/suction effect to attach to the box
    #                 suction_force, _, _ = skill_utils.calculate_suction(self.block_pos, self.robot_pos, self.num_envs, self.kp_suction, self.block_index, self.bodies_per_env)
    #                 # Apply suction/magnetic force
    #                 self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(torch.reshape(suction_force, (self.num_envs*self.bodies_per_env, 3))), None, gymapi.ENV_SPACE)

    #             # Step the similation
    #             sim_init.step(self.gym, self.sim)
    #             sim_init.refresh_states(self.gym, self.sim)

    #             # Step rendering and store data
    #             self.sim_time = np.append(self.sim_time, t_prev)
    #             t_now = time.monotonic()
    #             # print('Whole freq', format(1/(t_now-t_prev), '.2f'))
    #             if (t_now - t_prev) < params.dt:
    #                 sim_init.step_rendering(self.gym, self.sim, self.viewer, sync_frame_time=True)
    #             else:
    #                 sim_init.step_rendering(self.gym, self.sim, self.viewer, sync_frame_time=False)
    #             t_prev = t_now
    #             self.next_fps_report, self.frame_count, self.t1 = sim_init.time_logging(self.gym, self.sim, self.next_fps_report, self.frame_count, self.t1, self.num_envs, self.sim_time)

    # def destroy(self):
    #     # Destroy the simulation
    #     sim_init.destroy_sim(self.gym, self.sim, self.viewer)

if __name__== "__main__":
    run_sim()