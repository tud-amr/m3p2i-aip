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
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()

        # Helper variables, same as in fusion_mppi
        self.suction_active = params.suction_active
        self.block_index = params.block_index
        self.kp_suction = params.kp_suction

        # Time logging
        self.frame_count = 0
        self.next_fps_report = 2.0
        self.t1 = 0
        self.count = 0
        self.sim_time = np.array([])
        self.task_freq_array = np.array([])
        self.motion_freq_array = np.array([])

        # Set server address
        self.server_address = './uds_socket'

    def reset(self):
        # Press 'R' to reset the simulation
        reset_flag = False
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "reset" and evt.value > 0:
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_root_states))
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.initial_dof_states))
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
                message = s.recv(1024)

                # Send dof states to mppi and receive message
                s.sendall(data_transfer.torch_to_bytes(self.dof_states))
                message = s.recv(1024)

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

                self.action = actions[0]
                if self.robot == 'boxer':
                    r = 0.08
                    L = 2*0.157
                    # Diff drive fk
                    action_fk = self.action.clone()
                    action_fk[0] = (self.action[0] / r) - ((L*self.action[1])/(2*r))
                    action_fk[1] = (self.action[0] / r) + ((L*self.action[1])/(2*r))
                    self.action = action_fk

                if self.count == 0:
                    self.action_seq = torch.zeros_like(self.action)
                self.action_seq = torch.cat((self.action_seq, self.action), 0)
                
                # Apply optimal action
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

                # Step rendering
                self.sim_time = np.append(self.sim_time, t_prev)
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

        num_dof = int(list(self.action.size())[0])
        self.action_seq = self.action_seq.reshape(len(self.sim_time), num_dof)
        ctrl_input = np.zeros([len(self.sim_time), num_dof])

        fig, axs = plt.subplots(num_dof)
        fig.suptitle('Control Inputs')
        plot_colors = ['hotpink','darkviolet','mediumblue']

        if self.robot == "point_robot" or self.robot == "heijn":
            label = ['x_vel', 'y_vel', 'theta_vel']
        elif self.robot == "boxer":
            label = ['r_vel', 'l_vel']

        for j in range(num_dof):
            ctrl_input[:,j] = self.action_seq[:,j].tolist()
            axs[j].plot(self.sim_time, ctrl_input[:,j], color=plot_colors[j], marker=".")
            axs[j].legend([label[j]])
            axs[j].set(xlabel = 'Time [s]')

        print("Avg. simulation frequency ", format(len(self.action_seq)/self.sim_time[-1], '.2f'))
        print("Avg. task planner frequency", format(np.average(self.task_freq_array), '.2f'))
        print("Avg. motion planner frequency", format(np.average(self.motion_freq_array[np.nonzero(self.motion_freq_array)]), '.2f'))
        plt.show()

    def destroy(self):
        # Destroy the simulation
        sim_init.destroy_sim(self.gym, self.sim, self.viewer)

if __name__== "__main__":
    params = params_utils.load_params()
    sim = SIM(params)
    sim.run()
    sim.plot()
    sim.destroy()
