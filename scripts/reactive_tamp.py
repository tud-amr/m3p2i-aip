from isaacgym import gymtorch
from m3p2i_aip.planners.motion_planner import m3p2i
from m3p2i_aip.planners.task_planner import task_planner
from m3p2i_aip.utils import sim_init, data_transfer
from m3p2i_aip.params import params_utils
import torch, time, copy, socket, numpy as np
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)


class REACTIVE_TAMP:
    def __init__(self, params) -> None:
        # Make the environment and simulation
        self.params = params
        self.num_envs = params.num_envs
        self.is_mobile_robot = True if params.robot in ['point_robot', 'heijn', 'boxer'] else False
        self.gym, self.sim, self.viewer, _, _ = sim_init.make(params)

        # Acquire states
        states_dict = sim_init.acquire_states(self.gym, self.sim, self.params)
        self.dofs_per_robot = states_dict["dofs_per_robot"]
        self.robot_pos = states_dict["robot_pos"]
        self.block_state = states_dict["block_state"]
        self.cube_state = states_dict["cube_state"]
        self.cube_goal_state = states_dict["cube_goal_state"]
        self.ee_l_state = states_dict["ee_l_state"]
        self.ee_r_state = states_dict["ee_r_state"]

        # Choose the task planner
        if self.is_mobile_robot:
            self.task_planner = task_planner.PLANNER_SIMPLE(params.task, params.block_goal)
        else:
            self.task_planner = task_planner.PLANNER_AIF_PANDA()

        # Choose the motion planner
        self.motion_planner = m3p2i.M3P2I(self.params)
        self.motion_planner.set_mode(mppi_mode = 'halton-spline', # 'halton-spline', 'simple'
                                     sample_method = 'halton',    # 'halton', 'random'
                                     multi_modal = params.multimodal)
        self.prefer_pull = -1
        
        # Make sure the socket does not already exist
        self.server_address = './uds_socket'
        data_transfer.check_server(self.server_address)

    def tamp_interface(self, robot_pos, stay_still):
        # Update task and goal in the task planner
        start_time = time.monotonic()
        if self.is_mobile_robot:
            self.task_planner.update_plan(robot_pos, stay_still)
        else:
            self.task_planner.update_plan(self.cube_state[0, :7], 
                                          self.cube_goal_state[0, :7], 
                                          (self.ee_l_state[0, :7]+self.ee_r_state[0, :7])/2)
        self.task_freq = format(1/(time.monotonic()-start_time), '.2f')

        # Update params according to the plan
        self.params = self.task_planner.update_params(self.params)

        # Update task and goal in the motion planner
        # print('task:', self.task_planner.task, 'goal:', self.task_planner.curr_goal)
        self.motion_planner.update_task(self.task_planner.task, self.task_planner.curr_goal)

        # Update params in the motion planner
        self.params = self.motion_planner.update_params(self.params, self.prefer_pull)

        # Check task succeeds or not
        task_success = False
        if self.is_mobile_robot:
            task_success = self.task_planner.check_task_success(robot_pos, self.block_state[0, :])
        else:
            task_success = self.task_planner.check_task_success((self.ee_l_state[0, :7]+self.ee_r_state[0, :7])/2)
        task_success = task_success and not stay_still
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
                    conn.sendall(bytes(params.task, 'utf-8'))

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
                        actions = torch.zeros(self.motion_planner.u_per_command, self.motion_planner.nu, **params.tensor_args)
                        self.motion_freq = 0 # should be filtered later
                        self.prefer_pull=-1
                    # Compute optimal action and send to real simulator
                    else:
                        motion_time_prev = time.monotonic()
                        actions = self.motion_planner.command(s[0])
                        self.motion_freq = format(1/(time.monotonic()-motion_time_prev), '.2f')
                        self.prefer_pull = self.motion_planner.get_weights_preference()
                    conn.sendall(data_transfer.torch_to_bytes(actions))

                    # Send freq data
                    message = conn.recv(1024)
                    freq_data = np.array([self.motion_freq, self.params.suction_active], dtype = float)
                    conn.sendall(data_transfer.numpy_to_bytes(freq_data))

                    # Send top trajs
                    if self.motion_freq != 0:
                        print("Motion freq", self.motion_freq)
                        res = conn.recv(1024)
                        conn.sendall(data_transfer.torch_to_bytes(self.motion_planner.top_trajs))
                    print("Task succeeds!!") if task_success else False

if __name__== "__main__":
    params = params_utils.load_params()
    reactive_tamp = REACTIVE_TAMP(params)
    reactive_tamp.run()