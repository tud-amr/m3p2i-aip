from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from fusion_mppi import mppi, fusion_mppi
from utils import env_conf, sim_init, data_transfer
import time
import copy
import socket, io
import numpy as np
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

class PLANNER_PATROLLING:
    def __init__(self, goals) -> None:
        self.goals = goals
        self.goal_id = 0
        self.curr_goal = self.goals[self.goal_id]
        self.task = "navigation"
    
    def update_goal(self):
        self.goal_id += 1
        if self.goal_id >= self.goals.size(0):
            self.goal_id = 0
        self.curr_goal = self.goals[self.goal_id]


class REACTIVE_TAMP:
    def __init__(self) -> None:
        # Make the environment and simulation
        allow_viewer = False
        self.visualize_rollouts = False
        self.num_envs = 100
        spacing = 10.0
        self.robot = "point_robot"               # choose from "point_robot", "boxer", "albert"
        self.environment_type = "normal"         # choose from "normal", "battery"
        control_type = "vel_control"             # choose from "vel_control", "pos_control", "force_control"
        self.gym, self.sim, self.viewer, envs, _ = sim_init.make(allow_viewer, self.num_envs, spacing, self.robot, self.environment_type, control_type)

        # Acquire states
        self.dof_states, self.num_dofs, self.num_actors, self.root_states = sim_init.acquire_states(self.gym, self.sim, print_flag=False)
        actors_per_env = int(self.num_actors/self.num_envs)
        bodies_per_env = self.gym.get_env_rigid_body_count(envs[0])

        # Choose the task planner
        self.task_planner = PLANNER_PATROLLING(goals = torch.tensor([[-3, -3], [3, -3], [3, 3], [-3, 3]], device="cuda:0"))

        # Choose the motion planner
        self.motion_planner = fusion_mppi.FUSION_MPPI(
                            dynamics=None, 
                            running_cost=None, 
                            nx=2, 
                            noise_sigma = torch.tensor([[1, 0], [0, 1]], device="cuda:0", dtype=torch.float32),
                            num_samples=self.num_envs, 
                            horizon=20,
                            lambda_=0.1, 
                            device="cuda:0", 
                            u_max=torch.tensor([1.5, 1.5]),
                            u_min=torch.tensor([-1.5, -1.5]),
                            step_dependent_dynamics=True,
                            terminal_state_cost=None,
                            sample_null_action=True,
                            use_priors=False,
                            use_vacuum = False,
                            robot_type=self.robot,
                            u_per_command=20,
                            actors_per_env=actors_per_env,
                            env_type=self.environment_type,
                            bodies_per_env=bodies_per_env,
                            filter_u=True
                            )
        
        # Make sure the socket does not already exist
        self.server_address = './uds_socket'
        data_transfer.check_server(self.server_address)

    def tamp_interface(self, curr_pos):
        # Update task planner goal
        if torch.norm(curr_pos - self.task_planner.curr_goal) < 0.1:
            self.task_planner.update_goal()
        # Send task and goal to motion planner
        self.motion_planner.update_task(self.task_planner.task)
        self.motion_planner.update_nav_goal(self.task_planner.curr_goal)

    def run(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            # Build the connection
            s.bind(self.server_address)
            s.listen()
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                
                # Send info for simulation, robot and environment types
                res = conn.recv(1024)
                conn.sendall(self.robot.encode())

                res = conn.recv(1024)
                conn.sendall(self.environment_type.encode())

                i=0
                while True:
                    i+=1
                    # Receive dof states
                    res = conn.recv(2**14)
                    r = copy.copy(res)
                    _dof_states = data_transfer.bytes_to_torch(r).repeat(self.num_envs, 1)
                    conn.sendall(b"next please")

                    # Receive root states
                    res = conn.recv(2**14)
                    r = copy.copy(res)
                    _root_states = data_transfer.bytes_to_torch(r).repeat(self.num_envs, 1)

                    # Reset the simulator to requested state
                    s = _dof_states.view(-1, 4) # [x, v_x, y, v_y]
                    self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(s))
                    self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(_root_states))
                    sim_init.refresh_states(self.gym, self.sim)

                    # Update TAMP interface
                    self.tamp_interface(torch.tensor([s[0][0], s[0][2]], device="cuda:0"))

                    # Update gym in mppi
                    self.motion_planner.update_gym(self.gym, self.sim, self.viewer)

                    # Compute optimal action and send to real simulator
                    actions = self.motion_planner.command(s[0])
                    conn.sendall(data_transfer.torch_to_bytes(actions))

                    # Send rollouts data
                    res = conn.recv(1024)
                    conn.sendall(data_transfer.torch_to_bytes(int(self.visualize_rollouts)))
                    if self.visualize_rollouts:
                        # Get the rollouts trajectory
                        rollouts = self.motion_planner.states[0, :, :, :].cpu().clone().numpy()
                        current_traj = np.zeros((self.motion_planner.T, 2))
                        K = self.motion_planner.K
                        res = conn.recv(1024)
                        conn.sendall(data_transfer.numpy_to_bytes(self.motion_planner.K))
                        for i in range(K):
                            res4 = conn.recv(1024)
                            current_traj[:, 1] = rollouts[i][:, 0]     # x pos
                            current_traj[:, 0] = rollouts[i][:, 2]     # y pos
                            conn.sendall(data_transfer.numpy_to_bytes(current_traj))

reactive_tamp = REACTIVE_TAMP()
reactive_tamp.run()