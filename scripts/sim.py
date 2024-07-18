from isaacgym import gymapi, gymtorch
import torch, time, numpy as np, socket
from m3p2i_aip.params import params_utils
from m3p2i_aip.utils import sim_init, data_transfer, skill_utils
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

class SIM():
    def __init__(self, params) -> None:
        # Make the environment and simulation
        self.num_envs = params.sim_num_envs
        self.mobile_robot = True if params.robot in ['point_robot', 'heijn', 'boxer'] else False
        self.environment_type = params.environment_type
        self.gym, self.sim, self.viewer, self.envs, _ = sim_init.make(params.sim_allow_viewer, self.num_envs, params.spacing, params.robot, self.environment_type, dt = params.dt)

        # Acquire states
        states_dict = sim_init.acquire_states(self.gym, self.sim, params, "sim")
        self.root_states = states_dict["root_states"]
        self.dof_states = states_dict["dof_states"]
        self.num_actors = states_dict["num_actors"]
        self.root_states = states_dict["root_states"]
        self.bodies_per_env = states_dict["bodies_per_env"]
        self.block_pos = states_dict["block_pos"]
        self.robot_pos = states_dict["robot_pos"]
        self.robot_vel = states_dict["robot_vel"]
        self.block_state = states_dict["block_state"]
        self.dofs_per_robot = states_dict["dofs_per_robot"]
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()
        self.ee_l_state = states_dict["ee_l_state"]
        if self.environment_type == "normal":
            self.dyn_obs_pos = states_dict["dyn_obs_pos"]
            self.dyn_obs_pos_seq = self.dyn_obs_pos.clone()

        # Helper variables, same as in m3p2i
        self.suction_active = params.suction_active
        self.block_index = params.block_index
        self.kp_suction = params.kp_suction

        # Data logging
        self.frame_count = 0
        self.next_fps_report = 2.0
        self.t1 = 0
        self.count = 0
        self.sim_time = np.array([])
        self.task_freq_array = np.array([])
        self.motion_freq_array = np.array([])
        self.action_seq = torch.zeros(self.dofs_per_robot, device="cuda:0")
        self.robot_pos_seq = self.robot_pos.clone()
        self.block_pos_seq = self.block_pos.clone()
        self.suction_exist = []
        self.suction_not_exist = []
        self.prefer_pull = []
        self.dyn_obs_id = 5
        self.dyn_obs_coll = 0
        self.allow_save_data = False
        self.elapsed_time = 0
        # Set server address
        self.server_address = './uds_socket'

    def reset(self):
        reset_flag = False
        if self.environment_type == 'cube':
            cubeA_index = 3
            cubeB_index = 4
        elif self.environment_type == 'albert_arena':
            cubeA_index = 1
            cubeB_index = 2
        obs_index = 6
        x_pos = torch.tensor([0.03, 0, 0], dtype=torch.float32, device='cuda:0').repeat(self.num_envs)
        y_pos = torch.tensor([0, 0.03, 0], dtype=torch.float32, device='cuda:0').repeat(self.num_envs)
        z_pos = torch.tensor([0, 0, 0.03], dtype=torch.float32, device='cuda:0').repeat(self.num_envs)
        cube_targets = {'key_up':-y_pos, 'key_down':y_pos, 'key_left':x_pos, 'key_right':-x_pos}
        goal_targets = {'up':-y_pos, 'down':y_pos, 'left':x_pos, 'right':-x_pos}
        obs_targets = {'1':x_pos, '2':-x_pos, '3':-y_pos, '4':y_pos, '5':z_pos, '6':-z_pos}
        for evt in self.gym.query_viewer_action_events(self.viewer):
            # Press 'R' to reset the simulation
            if evt.action == 'reset' and evt.value > 0:
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_root_states))
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_dof_states))
                reset_flag = True
            # Press WASD and up,left,right,down to interact with the cubes
            elif self.environment_type in ['cube', 'albert_arena'] and evt.value > 0:
                if evt.action in ['key_up', 'key_down', 'key_left', 'key_right']:
                    self.root_states[cubeA_index, 0:3] += cube_targets[evt.action]
                if evt.action in ['up', 'down', 'left', 'right']:
                    self.root_states[cubeB_index, 0:3] += goal_targets[evt.action]
                if evt.action in ['1', '2', '3', '4', '5', '6'] and self.environment_type == 'cube':
                    self.root_states[obs_index, 0:3] += obs_targets[evt.action]
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        sim_init.step(self.gym, self.sim)
        sim_init.refresh_states(self.gym, self.sim)

        return reset_flag
    
    def check_contact_force(self):
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        net_cf = gymtorch.wrap_tensor(_net_cf)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        if torch.sum(torch.abs(net_cf[self.dyn_obs_id, :2])) > 0.001:
            self.dyn_obs_coll += 1

    def run(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(self.server_address)
            t_prev = time.monotonic()

            while self.viewer is None or not self.gym.query_viewer_has_closed(self.viewer):
                # Check collision
                self.check_contact_force()

                # Reset the simulation when pressing 'R'
                reset_flag = self.reset()
                s.sendall(data_transfer.numpy_to_bytes(int(reset_flag)))
                self.task = s.recv(1024).decode('utf-8')

                # Send dof states to mppi and receive message
                s.sendall(data_transfer.torch_to_bytes(self.dof_states))
                self.curr_planner_task = s.recv(1024).decode('utf-8')

                # Send root states and receive optimal actions
                s.sendall(data_transfer.torch_to_bytes(self.root_states))
                b = s.recv(2**14)
                actions = data_transfer.bytes_to_torch(b)

                # Receive freq data
                s.sendall(b"freq data")
                b = s.recv(1024)
                freq_data = data_transfer.bytes_to_numpy(b)
                self.task_freq_array = np.append(self.task_freq_array, freq_data[0])
                self.motion_freq_array = np.append(self.motion_freq_array, freq_data[1])
                self.suction_active = int(freq_data[2])
                self.curr_goal = np.array([freq_data[3], freq_data[4]])
                self.prefer_pull.append(freq_data[5])
                task_success = int(freq_data[6])
                if len(self.sim_time) > 0:
                    self.elapsed_time = self.sim_time[-1]-self.sim_time[0]
                    if int(self.elapsed_time*100) % 20 == 0:
                        print("Time:", format(self.elapsed_time, '.1f'))
                if task_success or self.elapsed_time >= 40:
                    pass
                    # self.destroy()

                # Clear lines at the beginning
                self.gym.clear_lines(self.viewer)
                
                # Visualize top trajs
                if freq_data[1] != 0:
                    s.sendall(b"Visualize trajs")
                    _top_trajs = s.recv(2**12)
                    top_trajs = data_transfer.bytes_to_torch(_top_trajs)
                    sim_init.visualize_toptrajs(self.gym, self.viewer, self.envs[0], top_trajs, self.mobile_robot)
                
                # For multi-modal mppi
                if self.environment_type == 'normal':
                    dir_robot_bloc = (self.robot_pos-self.block_pos).squeeze(0)
                    check = torch.sum(actions[0] * dir_robot_bloc).item()
                    dis = torch.linalg.norm(dir_robot_bloc)
                    self.suction_active = False
                    if dis < 0.6 and check > 0 and self.curr_planner_task in ['pull', 'hybrid']:
                        self.suction_active = True

                # Apply forward kikematics and optimal action
                self.action = skill_utils.apply_fk(params.robot, actions[0])
                self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self.action))

                if self.suction_active:  
                    # Simulation of a magnetic/suction effect to attach to the box
                    suction_force, _, _ = skill_utils.calculate_suction(self.block_pos, self.robot_pos, self.num_envs, self.kp_suction, self.block_index, self.bodies_per_env)
                    # Get flag for suction exist
                    suction_exist = suction_force.any().item()
                    self.suction_exist.append(suction_exist)
                    self.suction_not_exist.append(not suction_exist)
                    # Apply suction/magnetic force
                    self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(torch.reshape(suction_force, (self.num_envs*self.bodies_per_env, 3))), None, gymapi.ENV_SPACE)
                else:
                    self.suction_exist.append(False)
                    self.suction_not_exist.append(True)
                # Update movement of dynamic obstacle
                if self.environment_type == 'normal':
                    sim_init.update_dyn_obs(self.gym, self.sim, self.num_actors, self.num_envs, self.count)
                    self.count += 1

                # Step the similation
                sim_init.step(self.gym, self.sim)
                sim_init.refresh_states(self.gym, self.sim)

                # Debug
                if self.environment_type == 'cube':
                    ee_rot_matrix = skill_utils.quaternion_rotation_matrix(self.ee_l_state[:, 3:7])
                    ee_zaxis = ee_rot_matrix[:, :, 2]
                    # print('tilt value', format(ee_zaxis[0, 0].item(), '.2f'))
                    # print('tilt value', format(ee_zaxis[0, 2].item(), '.2f'))

                # Step rendering and store data
                self.sim_time = np.append(self.sim_time, t_prev)
                self.action_seq = torch.cat((self.action_seq, self.action), 0)
                self.robot_pos_seq = torch.cat((self.robot_pos_seq, self.robot_pos), 0)
                self.block_pos_seq = torch.cat((self.block_pos_seq, self.block_pos), 0)
                if self.environment_type == 'normal':
                    self.dyn_obs_pos_seq = torch.cat((self.dyn_obs_pos_seq, self.dyn_obs_pos), 0)
                t_now = time.monotonic()
                # print('Whole freq', format(1/(t_now-t_prev), '.2f'))
                if (t_now - t_prev) < params.dt:
                    sim_init.step_rendering(self.gym, self.sim, self.viewer, sync_frame_time=True)
                else:
                    sim_init.step_rendering(self.gym, self.sim, self.viewer, sync_frame_time=False)
                t_prev = t_now
                self.next_fps_report, self.frame_count, self.t1 = sim_init.time_logging(self.gym, self.sim, self.next_fps_report, self.frame_count, self.t1, self.num_envs, freq_data)

    def destroy(self):
        # Destroy the simulation
        sim_init.destroy_sim(self.gym, self.sim, self.viewer)

if __name__== "__main__":
    params = params_utils.load_params()
    sim = SIM(params)
    sim.run()
    sim.destroy()