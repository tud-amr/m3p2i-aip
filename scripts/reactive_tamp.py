from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from fusion_mppi import mppi, fusion_mppi
from active_inference import task_planner
from utils import env_conf, sim_init, data_transfer
from params import params_utils
import time
import copy
import socket, io
import numpy as np
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)
from plot import plot_class


class REACTIVE_TAMP:
    def __init__(self, params) -> None:
        # Make the environment and simulation
        self.params = params
        self.allow_viewer = params.allow_viewer
        self.visualize_rollouts = params.visualize_rollouts
        self.num_envs = params.num_envs
        self.spacing = params.spacing
        self.dt = params.dt
        self.robot = params.robot
        self.mobile_robot = True if self.robot in ['point_robot', 'heijn', 'boxer'] else False
        self.environment_type = params.environment_type
        self.gym, self.sim, self.viewer, envs, _ = sim_init.make(self.allow_viewer, self.num_envs, self.spacing, self.robot, self.environment_type, dt=self.dt)

        # Acquire states
        states_dict = sim_init.acquire_states(self.gym, self.sim, self.params)
        self.dofs_per_robot = states_dict["dofs_per_robot"]
        self.actors_per_env = states_dict["actors_per_env"]
        self.bodies_per_env = states_dict["bodies_per_env"]
        self.robot_pos = states_dict["robot_pos"]
        self.block_pos = states_dict["block_pos"]
        self.cube_state = states_dict["cube_state"]
        self.cube_goal_state = states_dict["cube_goal_state"]
        self.cube_goal_state_new = self.cube_goal_state[0, :7].clone()
        self.cube_goal_state_new[2] += 0.1
        self.ee_l_state = states_dict["ee_l_state"]
        self.ee_r_state = states_dict["ee_r_state"]

        # Choose the task planner
        self.task = params.task
        if self.task == "patrolling":
            self.task_planner = task_planner.PLANNER_PATROLLING(goals = [[-3, -3], [3, -3], [3, 3], [-3, 3]])
        elif self.task == "reactive":
            self.task_planner = task_planner.PLANNER_AIF()
            # start plotting battery level
            plot_class.start_dash_server()
        elif self.task == "simple":
            self.task_planner = task_planner.PLANNER_SIMPLE("pick", self.cube_goal_state_new)  # "hybrid", [-3.75, -3.75]

        # Choose the motion planner
        self.motion_planner = fusion_mppi.FUSION_MPPI(
                            params = self.params,
                            dynamics = None, 
                            running_cost = None, 
                            nx = params.nx, 
                            noise_sigma = params.noise_sigma,
                            num_samples = self.num_envs, 
                            horizon = params.horizon,
                            lambda_ = params.lambda_, 
                            device = params.device, 
                            u_max = params.u_max,
                            u_min = params.u_min,
                            step_dependent_dynamics = params.step_dependent_dynamics,
                            terminal_state_cost = params.terminal_state_cost,
                            sample_null_action = params.sample_null_action,
                            use_priors = params.use_priors,
                            use_vacuum = params.suction_active,
                            robot_type = self.robot,
                            u_per_command = params.u_per_command,
                            actors_per_env = self.actors_per_env,
                            env_type = self.environment_type,
                            bodies_per_env = self.bodies_per_env,
                            filter_u = params.filter_u
                            )
        self.motion_planner.set_mode(
            mppi_mode = 'halton-spline',     # 'halton-spline', 'simple'
            sample_method = 'halton'  # 'halton', 'random'
        )
        self.prefer_pull = -1
        
        # Make sure the socket does not already exist
        self.server_address = './uds_socket'
        data_transfer.check_server(self.server_address)

    def tamp_interface(self, robot_pos, stay_still):
        # Update task and goal in the task planner
        start_time = time.monotonic()
        self.task_planner.update_plan(robot_pos, stay_still)
        self.task_freq = format(1/(time.monotonic()-start_time), '.2f')

        # Update params according to the plan
        self.params = self.task_planner.update_params(self.params)

        # Update task and goal in the motion planner
        # print('task:', self.task_planner.task, 'goal:', self.task_planner.curr_goal)
        self.motion_planner.update_task(self.task_planner.task, self.task_planner.curr_goal)

        # Update params in the motion planner
        self.params = self.motion_planner.update_params(self.params, self.prefer_pull)

        # Check task succeeds or not
        if self.task_planner.task != 'pick':
            block_pose = self.block_pos[0, :]
        else:
            block_pose = self.cube_state[0, :3]
            norm = torch.norm(self.cube_goal_state_new - self.cube_state[0, :7])
            if norm < 0.1:
                self.task_planner.task = 'place'
                ee_goal = self.cube_goal_state_new.clone()
                ee_goal[2] += 0.3
                self.task_planner.curr_goal = ee_goal
        task_success = self.task_planner.check_task_success(robot_pos, block_pose)
        return task_success

    def reset(self, i, reset_flag):
        if reset_flag:
            self.task_planner.reset_plan()
            i = 0
        return i

    def run(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            # Build the connection
            s.bind(self.server_address)
            s.listen()
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                i=0
                while True:
                    i+=1
                    # Reset the plan when receiving the flag
                    res = conn.recv(1024)
                    reset_flag = data_transfer.bytes_to_numpy(res)
                    i = self.reset(i, reset_flag)
                    conn.sendall(bytes(self.task, 'utf-8'))

                    # Receive dof states
                    res = conn.recv(2**14)
                    r = copy.copy(res)
                    _dof_states = data_transfer.bytes_to_torch(r).repeat(self.num_envs, 1)
                    conn.sendall(bytes(self.task_planner.task, 'utf-8'))

                    # Receive root states
                    res = conn.recv(2**14)
                    r = copy.copy(res)
                    _root_states = data_transfer.bytes_to_torch(r).repeat(self.num_envs, 1)

                    # Reset the simulator to requested state
                    s = _dof_states.view(-1, 2*self.dofs_per_robot)
                    self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(s))
                    self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(_root_states))
                    sim_init.refresh_states(self.gym, self.sim)

                    # Update TAMP interface
                    stay_still = True if i < 50 else False
                    task_success = self.tamp_interface(self.robot_pos[0, :], stay_still)

                    # Update gym in mppi
                    self.motion_planner.update_gym(self.gym, self.sim, self.viewer)

                    # Stay still if the task planner has no task
                    if self.task_planner.task == "None" or stay_still or task_success:
                        actions = torch.zeros(self.motion_planner.u_per_command, self.motion_planner.nu, device="cuda:0")
                        self.motion_freq = 0 # should be filtered later
                        self.prefer_pull=-1
                    # Compute optimal action and send to real simulator
                    else:
                        motion_time_prev = time.monotonic()
                        actions = self.motion_planner.command(s[0])
                        motion_time_now = time.monotonic()
                        self.motion_freq = format(1/(motion_time_now-motion_time_prev), '.2f')
                        self.prefer_pull = self.motion_planner.get_weights_preference()
                    # print('Motion freq', self.motion_freq)
                    conn.sendall(data_transfer.torch_to_bytes(actions))

                    # Send freq data
                    message = conn.recv(1024)
                    # print(self.task_planner.curr_goal[0].item())
                    freq_data = np.array([self.task_freq, self.motion_freq, self.params.suction_active,
                                          self.task_planner.curr_goal[0].item(), self.task_planner.curr_goal[1].item(),
                                          self.prefer_pull], dtype = float)
                    conn.sendall(data_transfer.numpy_to_bytes(freq_data))

                    # Visualize rollouts
                    if self.visualize_rollouts and self.motion_freq != 0 and self.mobile_robot:
                        # Get the rollouts trajectory
                        rollouts = self.motion_planner.states.cpu().clone().numpy()
                        current_traj = np.zeros((self.motion_planner.T, 2))
                        K = self.motion_planner.K
                        res = conn.recv(1024)
                        conn.sendall(data_transfer.numpy_to_bytes(self.motion_planner.K))
                        for i in range(K):
                            res4 = conn.recv(1024)
                            current_traj[:, 1] = rollouts[i][:, 0]     # x pos
                            current_traj[:, 0] = rollouts[i][:, 2]     # y pos
                            conn.sendall(data_transfer.numpy_to_bytes(current_traj))

if __name__== "__main__":
    params = params_utils.load_params()
    reactive_tamp = REACTIVE_TAMP(params)
    reactive_tamp.run()