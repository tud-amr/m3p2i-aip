from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from fusion_mppi import mppi
from utils import env_conf, sim_init, data_transfer, skill_utils, path_utils
from params import params_utils
from npy_append_array import NpyAppendArray
import time, numpy as np
import socket
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)
import matplotlib.pyplot as plt

class SIM():
    def __init__(self, params) -> None:
        # Make the environment and simulation
        self.allow_viewer = params.sim_allow_viewer
        self.num_envs = params.sim_num_envs
        self.spacing = params.spacing
        self.robot = params.robot
        self.mobile_robot = True if self.robot in ['point_robot', 'heijn', 'boxer'] else False
        self.environment_type = params.environment_type
        self.dt = params.dt
        self.gym, self.sim, self.viewer, self.envs, _ = sim_init.make(self.allow_viewer, self.num_envs, self.spacing, self.robot, self.environment_type, dt = self.dt)

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

        # Helper variables, same as in fusion_mppi
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
        self.allow_save_data = True
        self.elapsed_time = 0
        # Set server address
        self.server_address = './uds_socket'

    def reset(self):
        reset_flag = False
        cubeA_index = 3
        cubeB_index = 4
        x_pos = torch.tensor([0.03, 0, 0], dtype=torch.float32, device='cuda:0').repeat(self.num_envs)
        y_pos = torch.tensor([0, 0.03, 0], dtype=torch.float32, device='cuda:0').repeat(self.num_envs)
        cube_targets = {'key_up':-y_pos, 'key_down':y_pos, 'key_left':x_pos, 'key_right':-x_pos}
        goal_targets = {'up':-y_pos, 'down':y_pos, 'left':x_pos, 'right':-x_pos}
        for evt in self.gym.query_viewer_action_events(self.viewer):
            # Press 'R' to reset the simulation
            if evt.action == 'reset' and evt.value > 0:
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_root_states))
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_dof_states))
                reset_flag = True
            # Press WASD and up,left,right,down to interact with the cubes
            elif self.environment_type == 'cube' and evt.value > 0:
                if evt.action in ['key_up', 'key_down', 'key_left', 'key_right']:
                    self.root_states[cubeA_index, 0:3] += cube_targets[evt.action]
                if evt.action in ['up', 'down', 'left', 'right']:
                    self.root_states[cubeB_index, 0:3] += goal_targets[evt.action]
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
    
    def save_data(self):
        save_time = np.array([time.time()])
        save_robot_pos = self.robot_pos[0].cpu().detach().numpy()
        save_robot_vel = self.robot_vel[0].cpu().detach().numpy()
        save_block_state = self.block_state[0].cpu().detach().numpy()
        save_metrics = np.array([self.avg_sim_freq, self.avg_task_freq, self.avg_mot_freq, 
                                 self.dyn_obs_coll, self.task_time])
        concatenate_array = np.concatenate((save_time, save_robot_pos, save_robot_vel, 
                                            save_block_state, self.curr_goal, save_metrics))
        file_path = path_utils.get_plot_path() +'/point/case4_halton_push_corner2.npy'
        with NpyAppendArray(file_path) as npaa:
            npaa.append(np.array([concatenate_array]))
        data = np.load(file_path, mmap_mode="r")
        print(data[-1, :])
        print(time.asctime(time.localtime(data[-1, 0])))

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
                    if int(self.elapsed_time) % 5 == 0:
                        print(int(self.elapsed_time))
                if task_success or self.elapsed_time >= 40:
                    if self.environment_type != 'cube':
                        # self.plot()
                        pass
                        # if self.allow_save_data:
                        #     self.save_data()
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
                dir_robot_bloc = (self.robot_pos-self.block_pos).squeeze(0)
                check = torch.sum(actions[0] * dir_robot_bloc).item()
                dis = torch.linalg.norm(dir_robot_bloc)
                self.suction_active = False
                if dis < 0.55 and check > 0 and self.curr_planner_task in ['pull', 'hybrid']:
                    self.suction_active = True
                # print(self.suction_active)
                # Apply forward kikematics and optimal action
                self.action = skill_utils.apply_fk(self.robot, actions[0])
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
                # ee_rot_matrix = skill_utils.quaternion_rotation_matrix(self.ee_l_state[:, 3:7])
                # ee_zaxis = ee_rot_matrix[:, :, 2]
                # print('tilt value', format(ee_zaxis[0, 0].item(), '.2f'))

                # Step rendering and store data
                self.sim_time = np.append(self.sim_time, t_prev)
                self.action_seq = torch.cat((self.action_seq, self.action), 0)
                self.robot_pos_seq = torch.cat((self.robot_pos_seq, self.robot_pos), 0)
                self.block_pos_seq = torch.cat((self.block_pos_seq, self.block_pos), 0)
                if self.environment_type == 'normal':
                    self.dyn_obs_pos_seq = torch.cat((self.dyn_obs_pos_seq, self.dyn_obs_pos), 0)
                t_now = time.monotonic()
                # print('Whole freq', format(1/(t_now-t_prev), '.2f'))
                if (t_now - t_prev) < self.dt:
                    sim_init.step_rendering(self.gym, self.sim, self.viewer, sync_frame_time=True)
                else:
                    sim_init.step_rendering(self.gym, self.sim, self.viewer, sync_frame_time=False)
                t_prev = t_now
                self.next_fps_report, self.frame_count, self.t1 = sim_init.time_logging(self.gym, self.sim, self.next_fps_report, self.frame_count, self.t1, self.num_envs, freq_data)

    def plot(self):
        # Saving and plotting
        self.sim_time-= self.sim_time[0]
        self.sim_time = np.append(0, self.sim_time)
        ctrl_input = self.action_seq.reshape(len(self.sim_time), self.dofs_per_robot).cpu().numpy()
        robot_pos_array = self.robot_pos_seq.cpu().numpy()
        block_pos_array = self.block_pos_seq.cpu().numpy()
        robot_to_block = robot_pos_array - block_pos_array
        block_to_goal = self.curr_goal - block_pos_array
        robot_to_goal = robot_pos_array - self.curr_goal
        robot_to_block_dist = np.linalg.norm(robot_to_block, axis=1)
        block_to_goal_dist = np.linalg.norm(block_to_goal, axis=1) 
        robot_to_goal_dist = np.linalg.norm(robot_to_goal, axis=1)
        cos_theta = np.sum(robot_to_block*block_to_goal, 1)/(robot_to_block_dist*block_to_goal_dist)
        self.suction_exist.insert(0, False)
        self.suction_not_exist.insert(0, True)
        self.prefer_pull.insert(0, -1)
        robot_block_close = robot_to_block_dist <= 0.5
        robot_block_not_close = robot_to_block_dist > 0.5
        if self.curr_planner_task in ['navigation', 'go_recharge']:
            draw_block = False
        elif self.curr_planner_task in ['push', 'pull', 'hybrid']:
            draw_block = True
            dyn_obs_pos_array = self.dyn_obs_pos_seq.cpu().numpy()
            rob_to_dyn_obs = robot_pos_array - dyn_obs_pos_array
            rob_to_dyn_obs_dist = np.linalg.norm(rob_to_dyn_obs, axis=1)
            block_to_dyn_obs = block_pos_array - dyn_obs_pos_array
            block_to_dyn_obs_dist = np.linalg.norm(block_to_dyn_obs, axis=1)

        if not self.allow_save_data:
            # Draw the control inputs
            fig1, axs1 = plt.subplots(self.dofs_per_robot)
            fig1.suptitle('Control Inputs')
            plot_colors = ['hotpink','darkviolet','mediumblue', 'red']
            if self.robot in ['point_robot', 'heijn']:
                label = ['x_vel', 'y_vel', 'theta_vel']
            elif self.robot == 'boxer':
                label = ['r_vel', 'l_vel']
            for j in range(self.dofs_per_robot):
                axs1[j].plot(self.sim_time, ctrl_input[:,j], color=plot_colors[j], marker=".")
                axs1[j].legend([label[j]])
                axs1[j].set_ylabel('v [m/s]', rotation=0)
            axs1[-1].set(xlabel = 'Time [s]')

            # Draw the pos of robot and block
            fig2, axs2 = plt.subplots(2)
            fig2.suptitle('Position')
            label_pos = ['x [m]', 'y [m]']
            for i in range(2):
                axs2[i].plot(self.sim_time, robot_pos_array[:, i], color=plot_colors[0], marker=".", label='robot')
                if draw_block:
                    axs2[i].plot(self.sim_time, block_pos_array[:, i], color=plot_colors[1], marker=".", label='block')
                    axs2[i].plot(self.sim_time, dyn_obs_pos_array[:, i], color=plot_colors[2], marker=".", label='obstacle')
                axs2[i].set_ylabel(label_pos[i], rotation=0)
            axs2[-1].set(xlabel = 'Time [s]')
            plt.legend()

            # Draw the distance
            if draw_block:
                fig3, axs3 = plt.subplots(2)
                fig3.suptitle('Distance')
                label_dis = ['robot_to_block', 'block_to_goal', 'robot_to_obstacle', 'block_to_obstacle']
                axs3[0].plot(self.sim_time, robot_to_block_dist, color=plot_colors[0], marker=".", label=label_dis[0])
                axs3[0].set_ylabel('[m]', rotation=0)
                axs3[0].plot(self.sim_time, block_to_goal_dist, color=plot_colors[1], marker=".", label=label_dis[1])
                axs3[0].legend()
                axs3[1].plot(self.sim_time, rob_to_dyn_obs_dist, color=plot_colors[2], marker=".", label=label_dis[2])
                axs3[1].plot(self.sim_time, block_to_dyn_obs_dist, color=plot_colors[3], marker=".", label=label_dis[3])
                axs3[1].set_ylabel('[m]', rotation=0)
                plt.axhline(y = 0.4, color = 'g', linestyle = '-')
                axs3[1].set_ylim(0, None)
                axs3[1].set_xlabel('Time [s]')
                axs3[1].legend()
            else:
                fig, ax = plt.subplots()
                fig.suptitle('Distance')
                ax.plot(self.sim_time, robot_to_goal_dist, color=plot_colors[0], marker=".")
                ax.legend('robot_to_goal')
                ax.set_ylabel('[m]', rotation=0)
                ax.set_xlabel('Time [s]')

            # Draw the trajectory
            fig4, axs4 = plt.subplots()
            fig4.suptitle('Trajectory')
            for i in range(robot_pos_array.shape[0]):
                circle_rob = plt.Circle((robot_pos_array[i, 0], robot_pos_array[i, 1]), 0.4, color='tomato', fill=False)
                axs4.add_patch(circle_rob)
                if draw_block:
                    circle_blo = plt.Circle((block_pos_array[i, 0], block_pos_array[i, 1]), 0.2, color='deepskyblue', fill=False)
                    axs4.add_patch(circle_blo)
            axs4.plot(robot_pos_array[:, 0], robot_pos_array[:, 1], 'o', color='r', markersize=0.6, label='robot')
            if draw_block:
                axs4.plot(block_pos_array[:, 0], block_pos_array[:, 1], 'o', color='b', markersize=0.6, label='block')
            axs4.plot(0, 0, "D", color='black', label='start')
            axs4.plot(self.curr_goal[0], self.curr_goal[1], "X", color='green', label='goal')
            axs4.set_xlabel('x [m]')
            axs4.set_ylabel('y [m]', rotation=0)
            axs4.axis('equal')
            plt.legend()

            # Draw the cos_theta
            if draw_block:
                fig5, axs5 = plt.subplots()
                fig5.suptitle('Cos(theta)')
                axs5.plot(self.sim_time, cos_theta, color='gray', marker=".", markersize=0.2)
                axs5.scatter(self.sim_time[robot_block_close*self.suction_exist], cos_theta[robot_block_close*self.suction_exist], marker='*', color='r', label='suction')
                axs5.scatter(self.sim_time[robot_block_close*self.suction_not_exist], cos_theta[robot_block_close*self.suction_not_exist], color='b', label='no suction')
                axs5.scatter(self.sim_time[robot_block_not_close], cos_theta[robot_block_not_close], marker='v', color='lime', label='approaching')
                axs5.set_xlabel('Time [s]')
                plt.legend()
            
            # Check weights distribution
            if self.curr_planner_task == 'hybrid':
                fig6, axs6 = plt.subplots()
                fig6.suptitle('Check weights distribution')
                axs6.scatter(self.sim_time, self.suction_exist, color='r', label='suction')
                axs6.scatter(self.sim_time, np.array(self.prefer_pull)-0.1, color='b', label='weight')
                axs6.set_xlabel('Time [s]')
                plt.legend()

        # Calculate metrics
        self.avg_sim_freq = len(self.sim_time)/self.sim_time[-1]
        self.avg_task_freq = np.average(self.task_freq_array)
        self.avg_mot_freq = np.average(self.motion_freq_array[np.nonzero(self.motion_freq_array)])
        robot_path_array = robot_pos_array[1:, :] - robot_pos_array[:-1, :]
        block_path_array = block_pos_array[1:, :] - block_pos_array[:-1, :]
        self.robot_path_length = np.sum(np.linalg.norm(robot_path_array, axis=1))
        self.block_path_length = np.sum(np.linalg.norm(block_path_array, axis=1))
        task_start_time = self.sim_time[np.nonzero(self.motion_freq_array)[0][0]]
        self.task_time = self.sim_time[-1]-task_start_time
        print('Avg. simulation frequency', format(self.avg_sim_freq, '.2f'))
        print('Avg. task planner frequency', format(self.avg_task_freq, '.2f'))
        print('Avg. motion planner frequency', format(self.avg_mot_freq, '.2f'))
        print('Robot path length', format(self.robot_path_length, '.2f'))
        print('Block path length', format(self.block_path_length, '.2f'))
        print('Dynamic obstacle collision times', self.dyn_obs_coll)
        print('Task completion time', format(self.task_time, '.2f'))
        plt.show()

    def destroy(self):
        # Destroy the simulation
        sim_init.destroy_sim(self.gym, self.sim, self.viewer)

if __name__== "__main__":
    params = params_utils.load_params()
    sim = SIM(params)
    sim.run()
    sim.destroy()