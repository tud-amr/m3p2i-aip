from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from fusion_mppi import mppi
from utils import env_conf, sim_init, data_transfer, skill_utils
from params import params_utils
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
        self.environment_type = params.environment_type
        self.dt = params.dt
        self.visualize_rollouts = params.visualize_rollouts
        self.gym, self.sim, self.viewer, self.envs, _ = sim_init.make(self.allow_viewer, self.num_envs, self.spacing, self.robot, self.environment_type, dt = self.dt)

        # Acquire states
        states_dict = sim_init.acquire_states(self.gym, self.sim, params, "sim")
        self.dof_states = states_dict["dof_states"]
        self.num_actors = states_dict["num_actors"]
        self.root_states = states_dict["root_states"]
        self.bodies_per_env = states_dict["bodies_per_env"]
        self.block_pos = states_dict["block_pos"]
        self.robot_pos = states_dict["robot_pos"]
        self.robot_vel = states_dict["robot_vel"]
        self.dofs_per_robot = states_dict["dofs_per_robot"]
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()

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

        # Set server address
        self.server_address = './uds_socket'

    def reset(self):
        # Press 'R' to reset the simulation
        reset_flag = False
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "reset" and evt.value > 0:
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_root_states))
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_dof_states))
                sim_init.step(self.gym, self.sim)
                sim_init.refresh_states(self.gym, self.sim)
                reset_flag = True
        return reset_flag

    def run(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(self.server_address)
            t_prev = time.monotonic()

            while self.viewer is None or not self.gym.query_viewer_has_closed(self.viewer):
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

                # Clear lines at the beginning
                self.gym.clear_lines(self.viewer)
                
                # Visualize rollouts
                if self.visualize_rollouts:
                    s.sendall(b"Visualize rollouts")
                    K = s.recv(1024)
                    K = int(data_transfer.bytes_to_numpy(K))
                    rollout_state = np.zeros((1, 2), dtype=np.float32)
                    for i in range(K):
                        s.sendall(b"next")
                        _rollout_state = s.recv(2**18)
                        rollout_state = data_transfer.bytes_to_numpy(_rollout_state)
                        sim_init.visualize_rollouts(self.gym, self.viewer, self.envs[0], rollout_state)

                # Visualize optimal trajectory
                #sim_init.visualize_traj(gym, viewer, envs[0], actions, dof_states)
                
                # Apply forward kikematics and optimal action
                self.action = skill_utils.apply_fk(self.robot, actions[0])
                self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self.action))

                if self.suction_active:  
                    pos_dir = self.block_pos - self.robot_pos
                    # True means the velocity moves towards block, otherwise means pull direction
                    flag_towards_block = torch.sum(self.robot_vel*pos_dir, 1) > 0
                    # simulation of a magnetic/suction effect to attach to the box
                    suction_force, _, _ = skill_utils.calculate_suction(self.block_pos, self.robot_pos, self.num_envs, self.kp_suction, self.block_index, self.bodies_per_env)
                    # print(suction_force)
                    # Set no suction force if robot moves towards the block
                    suction_force[flag_towards_block] = 0
                    # Apply suction/magnetic force
                    self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(torch.reshape(suction_force, (self.num_envs*self.bodies_per_env, 3))), None, gymapi.ENV_SPACE)

                # Update movement of dynamic obstacle
                if self.environment_type == 'normal':
                    sim_init.update_dyn_obs(self.gym, self.sim, self.num_actors, self.num_envs, self.count)
                    self.count += 1

                # Step the similation
                sim_init.step(self.gym, self.sim)
                sim_init.refresh_states(self.gym, self.sim)

                # Step rendering and store data
                self.sim_time = np.append(self.sim_time, t_prev)
                self.action_seq = torch.cat((self.action_seq, self.action), 0)
                self.robot_pos_seq = torch.cat((self.robot_pos_seq, self.robot_pos), 0)
                self.block_pos_seq = torch.cat((self.block_pos_seq, self.block_pos), 0)
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
        robot_to_block = np.linalg.norm(robot_pos_array - block_pos_array, axis=1)
        block_to_goal = np.linalg.norm(block_pos_array - self.curr_goal, axis=1)
        robot_to_goal = np.linalg.norm(robot_pos_array - self.curr_goal, axis=1)
        if self.curr_planner_task in ['navigation', 'go_recharge']:
            draw_block = False
        elif self.curr_planner_task in ['push', 'pull', 'hybrid']:
            draw_block = True

        # Draw the control inputs
        fig1, axs1 = plt.subplots(self.dofs_per_robot)
        fig1.suptitle('Control Inputs')
        plot_colors = ['hotpink','darkviolet','mediumblue']
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
        fig2.suptitle('Robot and Block Pos')
        label_pos = ['x [m]', 'y [m]']
        for i in range(2):
            axs2[i].plot(self.sim_time, robot_pos_array[:, i], color=plot_colors[0], marker=".", label='robot')
            if draw_block:
                axs2[i].plot(self.sim_time, block_pos_array[:, i], color=plot_colors[1], marker=".", label='block')
            axs2[i].set_ylabel(label_pos[i], rotation=0)
        axs2[-1].set(xlabel = 'Time [s]')
        plt.legend()

        # Draw the distance
        if draw_block:
            fig3, axs3 = plt.subplots(2)
            fig3.suptitle('Distance')
            label_dis = ['robot_to_block', 'block_to_goal']
            axs3[0].plot(self.sim_time, robot_to_block, color=plot_colors[0], marker=".")
            axs3[0].legend([label_dis[0]])
            axs3[0].set_ylabel('[m]', rotation=0)
            axs3[1].plot(self.sim_time, block_to_goal, color=plot_colors[1], marker=".")
            axs3[1].legend([label_dis[1]])
            axs3[1].set_ylabel('[m]', rotation=0)
            axs3[1].set_xlabel('Time [s]')
        else:
            fig, ax = plt.subplots()
            fig.suptitle('Distance')
            ax.plot(self.sim_time, robot_to_goal, color=plot_colors[0], marker=".")
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
        plt.show()

        # Calculate metrics
        print('Avg. simulation frequency', format(len(self.action_seq)/self.sim_time[-1], '.2f'))
        print('Avg. task planner frequency', format(np.average(self.task_freq_array), '.2f'))
        print('Avg. motion planner frequency', format(np.average(self.motion_freq_array[np.nonzero(self.motion_freq_array)]), '.2f'))
        robot_path_array = robot_pos_array[1:, :] - robot_pos_array[:-1, :]
        block_path_array = block_pos_array[1:, :] - block_pos_array[:-1, :]
        robot_path_length = np.sum(np.linalg.norm(robot_path_array, axis=1))
        block_path_length = np.sum(np.linalg.norm(block_path_array, axis=1))
        print('Robot path length', format(robot_path_length, '.2f'))
        print('Block path length', format(block_path_length, '.2f'))

    def destroy(self):
        # Destroy the simulation
        sim_init.destroy_sim(self.gym, self.sim, self.viewer)

if __name__== "__main__":
    params = params_utils.load_params()
    sim = SIM(params)
    sim.run()
    sim.plot()
    sim.destroy()