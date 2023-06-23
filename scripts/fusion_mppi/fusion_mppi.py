import fusion_mppi.mppi as mppi
import torch, math
import time
from isaacgym import gymtorch, gymapi, torch_utils
from utils import sim_init, skill_utils

class FUSION_MPPI(mppi.MPPI):
    def __init__(self, params, dynamics, running_cost, nx, noise_sigma, num_samples=100, horizon=15, device="cpu", 
                    terminal_state_cost=None, 
                    lambda_=1, 
                    noise_mu=None, 
                    u_min=None, 
                    u_max=None, 
                    u_init=None, 
                    U_init=None, 
                    u_scale=1, 
                    u_per_command=1, 
                    step_dependent_dynamics=False, 
                    rollout_samples=1, 
                    rollout_var_cost=0, 
                    rollout_var_discount=0.95, 
                    sample_null_action=False, 
                    use_priors=False,
                    use_vacuum = False,
                    robot_type='point_robot',
                    noise_abs_cost=False,
                    actors_per_env=None, 
                    env_type="normal",
                    bodies_per_env=None,
                    filter_u=True):
        super().__init__(params, dynamics, running_cost, nx, noise_sigma, num_samples, horizon, device, 
                    terminal_state_cost, 
                    lambda_, 
                    noise_mu, 
                    u_min, 
                    u_max, 
                    u_init, 
                    U_init, 
                    u_scale, 
                    u_per_command, 
                    step_dependent_dynamics, 
                    rollout_samples, 
                    rollout_var_cost, 
                    rollout_var_discount, 
                    sample_null_action, 
                    use_priors,
                    use_vacuum,
                    robot_type,
                    noise_abs_cost,
                    actors_per_env,
                    env_type, 
                    bodies_per_env,
                    filter_u)
        self.gym = None
        self.sim = None
        self.num_envs = num_samples
        self.robot = robot_type
        self.kp_suction = 400
        self.suction_active = use_vacuum
        self.bodies_per_env = bodies_per_env
        self.actors_per_env = actors_per_env
        self.env_type = env_type
        self.device = device
        
        # Additional variables for the environment
        if self.env_type == "normal":
            self.block_index = 7   # Pushing purple blox, index according to simulation
        if self.env_type == "lab":
            self.block_index = 4  
        self.block_goal = torch.tensor([0, 0], device="cuda:0")
        self.block_not_goal = torch.tensor([-2, 1], device="cuda:0")
        self.nav_goal = torch.tensor([3, 3], device="cuda:0")
        self.task = "navigation"  # "navigation", "push", "pull", "push_not_goal"
        self.align_weight = {"heijn":1, "point_robot":0.5, "boxer":1}
        self.align_offset = {"heijn":0.1, "point_robot":0.05}
        self.goal_quaternion = torch.tensor([0, 0, 0, 1], device="cuda:0").repeat(self.num_envs).view(self.num_envs, 4)

    def update_gym(self, gym, sim, viewer=None):
        self.gym = gym
        self.sim = sim
        self.viewer = viewer

        self.flag = True
        # Acquire states
        if self.flag:
            states_dict = sim_init.acquire_states(self.gym, self.sim, self.params)
            self.dof_states = states_dict["dof_states"]
            self.num_actors = states_dict["num_actors"]
            self.root_states = states_dict["root_states"]
            self.shaped_root_states = states_dict["shaped_root_states"]
            self.actors_per_env = states_dict["actors_per_env"]
            self.bodies_per_env = states_dict["bodies_per_env"]
            self.robot_states = states_dict["robot_states"]
            self.block_pos = states_dict["block_pos"]
            self.block_quat = states_dict["block_quat"]
            self.robot_pos = states_dict["robot_pos"]
            self.robot_vel = states_dict["robot_vel"]
            self.dyn_obs_pos = states_dict["dyn_obs_pos"]
            self.dyn_obs_vel = states_dict["dyn_obs_vel"]
            self.cube_state = states_dict["cube_state"]
            self.ee_l_state = states_dict["ee_l_state"]
            self.ee_r_state = states_dict["ee_r_state"]
            self.flag = False
    
    def update_task(self, task, goal):
        self.task = task
        if self.task in ['navigation', 'go_recharge']:
            self.nav_goal = goal
        elif self.task in ['push', 'pull', 'hybrid']:
            self.block_goal = goal
        elif self.task == 'pick':
            self.cube_goal_state = goal
        elif self.task == 'place':
            self.ee_goal = goal
    
    def update_params(self, params, weight_prefer_pull):
        self.params = params
        if self.task == 'hybrid' and weight_prefer_pull == 1:
            params.suction_active = True
        else:
            self.suction_active = params.suction_active
        return params

    def get_weights_preference(self):
        if self.task == 'hybrid':
            weight_push = torch.sum(self.weights[:int(self.num_envs/2)]).item()
            weight_pull = torch.sum(self.weights[int(self.num_envs/2):]).item()
            return int(weight_pull > weight_push)
        else:
            return -1

    def get_navigation_cost(self):
        return torch.clamp(torch.linalg.norm(self.robot_pos - self.nav_goal, axis=1)-0.05, min=0, max=1999) 
    
    def calculate_dist(self):
        self.robot_to_block = self.robot_pos - self.block_pos
        self.block_to_goal = self.block_goal - self.block_pos

        self.robot_to_block_dist = torch.linalg.norm(self.robot_to_block, axis = 1)
        self.block_to_goal_dist = torch.linalg.norm(self.block_to_goal, axis = 1)
        
        self.dist_cost = self.robot_to_block_dist + self.block_to_goal_dist * 5
        self.cos_theta = torch.sum(self.robot_to_block*self.block_to_goal, 1)/(self.robot_to_block_dist*self.block_to_goal_dist)

    def get_push_cost(self, hybrid):
        # Calculate dist cost
        self.calculate_dist()

        # Force the robot behind block and goal, align_cost is actually cos(theta)+1
        align_cost = self.align_weight[self.robot] * (self.cos_theta + 1) * 5
        # print('push align', align_cost[:10])
        # if self.robot != 'boxer':
        #     align_cost += torch.abs(self.robot_to_goal_dist - self.block_to_goal_dist - self.align_offset[self.robot])
        ori_cost = skill_utils.get_general_ori_cube2goal(self.block_quat, self.goal_quaternion)
        if hybrid:
            return self.dist_cost # [num_envs]
        else:
            return self.dist_cost + 3 * align_cost + 10 * ori_cost# [num_envs]
    
    def get_pull_cost(self, hybrid):
        pos_dir = self.block_pos - self.robot_pos
        # True means the velocity moves towards block, otherwise means pull direction
        flag_towards_block = torch.sum(self.robot_vel*pos_dir, 1) > 0

        # simulation of a magnetic/suction effect to attach to the box
        suction_force, dir, mask = skill_utils.calculate_suction(self.block_pos, self.robot_pos, self.num_envs, self.kp_suction, self.block_index, self.bodies_per_env)
        # Set no suction force if robot moves towards the block
        suction_force[flag_towards_block] = 0
        if hybrid:
            suction_force[:int(self.num_envs/2)] = 0
        # Apply suction/magnetic force
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(torch.reshape(suction_force, (self.num_envs*self.bodies_per_env, 3))), None, gymapi.ENV_SPACE)

        # Calculate dist cost
        self.calculate_dist()

        # Force the robot to be in the middle between block and goal, align_cost is actually 1-cos(theta)
        align_cost = (1 - self.cos_theta) * 5
        # print('pull align', align_cost[-10:])

        # Add the cost when the robot is close to the block and moves towards the block
        vel_cost = torch.zeros(self.num_envs, device="cuda:0")
        robot_block_close = self.robot_to_block_dist <= 0.5
        vel_cost[flag_towards_block*robot_block_close] = 0.5

        ori_cost = skill_utils.get_general_ori_cube2goal(self.block_quat, self.goal_quaternion)
        if hybrid:
            return self.dist_cost + vel_cost # [num_envs]
        else:
            return self.dist_cost + vel_cost + 3 * align_cost + 10 * ori_cost # [num_envs]

    def get_push_not_goal_cost(self):
        non_goal_cost = torch.clamp((1/torch.linalg.norm(self.block_not_goal - self.block_pos,axis = 1)), min=0, max=10)
        return torch.linalg.norm(self.robot_pos - self.block_pos, axis = 1) + non_goal_cost

    def _predict_dyn_obs(self, factor, t):
        # Obs boundary [-2.5, 1.5] <--> [-1.5, 2.5]
        obs_lb = torch.tensor([-2.5, 1.5], dtype=torch.float32, device="cuda:0")
        obs_ub = torch.tensor([-1.5, 2.5], dtype=torch.float32, device="cuda:0")
        self.dyn_obs_vel = torch.clamp(self.dyn_obs_vel, min = -0.001, max = 0.001)
        pred_pos = self.dyn_obs_pos + t * self.dyn_obs_vel * 10
        # Check the prec_pos and boundary
        exceed_ub = pred_pos[:, 1] > obs_ub[1]
        exceed_lb = pred_pos[:, 1] < obs_lb[1]
        pred_pos[exceed_ub] = 2 * obs_ub - pred_pos[exceed_ub]
        pred_pos[exceed_lb] = 2 * obs_lb - pred_pos[exceed_lb]
        # Compute the cost
        dyn_obs_cost = factor * torch.exp(-torch.norm(pred_pos - self.robot_pos, dim=1))

        return dyn_obs_cost

    def get_panda_pick_cost(self, hybrid):
        self.ee_state = (self.ee_l_state + self.ee_r_state) / 2
        reach_cost = torch.linalg.norm(self.ee_state[:,:3] - self.cube_state[:,:3], axis = 1) 
        goal_cost = torch.linalg.norm(self.cube_goal_state[:3] - self.cube_state[:,:3], axis = 1) #+ 2*torch.abs(self.block_goal[2] - block_state[:,2])
        # Close the gripper when close to the cube
        gripper_dist = torch.linalg.norm(self.ee_l_state[:, :3] - self.ee_r_state[:, :3], axis=1)
        gripper_cost = 2 * (1 - gripper_dist)
        gripper_cost[reach_cost < 0.1] = 0

        if self.robot == 'omni_panda':
            # get jacobian tensor
            self.gym.refresh_jacobian_tensors(self.sim)
            # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
            _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
            jacobian = gymtorch.wrap_tensor(_jacobian)
            # jacobian entries corresponding to franka hand 
            j_eef = jacobian[:, 11, :, 3:10]
            A = torch.bmm(j_eef, torch.transpose(j_eef, 1,2))
            eig = torch.real(torch.linalg.eigvals(A))
            manip_cost = torch.sqrt(torch.max(eig, dim = 1)[0]/torch.min(eig, dim = 1)[0])
            manip_cost = torch.nan_to_num(manip_cost, nan=500)
        else:
            manip_cost = torch.zeros_like(reach_cost)
        
        # Compute the orientation cost
        cube_quaternion = self.cube_state[:, 3:7]
        goal_quatenion = self.cube_goal_state[3:7].repeat(self.num_envs).view(self.num_envs, 4)
        # To make the cube fit the goal's orientation well
        ori_cube2goal = skill_utils.get_general_ori_cube2goal(cube_quaternion, goal_quatenion) 
        ori_cost = 3 * ori_cube2goal

        # Compute the tilt value between ee and cube
        tilt_cost = self.get_pick_tilt_cost(hybrid)
        tilt_cost[reach_cost<=0.05] = 0
        # print('reach', reach_cost)
        # print('griper dist', gripper_dist)
        total_cost = 0.2 * manip_cost + 10 * reach_cost + 10 * goal_cost + ori_cost + gripper_cost + tilt_cost

        return  total_cost #+ align_cost multiply 10*reach_cost when using mppi_mode == storm

    def get_pick_tilt_cost(self, hybrid):
        # This measures the cost of the tilt angle between the end effector and the cube
        ee_quaternion = self.ee_l_state[:, 3:7]
        cube_quaternion = self.cube_state[:, 3:7]
        if not hybrid:
            # To make the z-axis direction of end effector to be perpendicular to the cube surface
            ori_ee2cube = skill_utils.get_general_ori_ee2cube(ee_quaternion, cube_quaternion, tilt_value=0.4)
        else:
            # To combine costs of different tilt angles
            batch_num = int(self.num_envs/2)
            cost_1 = skill_utils.get_general_ori_ee2cube(ee_quaternion[:batch_num], 
                                                         cube_quaternion[:batch_num], tilt_value=0) # 0.2
            cost_2 = skill_utils.get_general_ori_ee2cube(ee_quaternion[batch_num:2*batch_num], 
                                                         cube_quaternion[batch_num:2*batch_num], tilt_value=0.7) #0.9
            ori_ee2cube =  torch.cat((cost_1, cost_2), dim=0)

        return 3 * ori_ee2cube

    def get_panda_place_cost(self):
        gripper_dist = torch.linalg.norm(self.ee_l_state[:, :3] - self.ee_r_state[:, :3], axis=1)
        gripper_cost = 1 - gripper_dist
        self.ee_state = (self.ee_l_state + self.ee_r_state) / 2
        reach_cost = torch.linalg.norm(self.ee_state[:,:7] - self.ee_goal[:7], axis=1)
        # If gripper is not fylly open, no reach cost
        reach_cost[gripper_dist <= 0.078] = 0
        # If gripper is fully open, no gripper cost, retract the arm
        gripper_cost[gripper_dist > 0.078] = 0
        return 10 * gripper_cost + 10 * reach_cost

    @mppi.handle_batch_input
    def _dynamics(self, state, u, t):
        # Use inverse kinematics if the MPPI action space is different than dof velocity space
        u_ = skill_utils.apply_ik(self.robot, u)
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(u_))
        
        time_1 = time.monotonic()
        # Step the simulation
        sim_init.step(self.gym, self.sim)  # very essential 0.002s

        time_2 = time.monotonic()
        gap_2 = format(time_2-time_1, '.5f')
        # print('very gap 2', gap_2) # 

        sim_init.refresh_states(self.gym, self.sim)
        sim_init.step_rendering(self.gym, self.sim, self.viewer, sync_frame_time=True)

        # Return the current states
        states = torch.stack([self.robot_pos[:, 0], 
                              self.robot_vel[:, 0], 
                              self.robot_pos[:, 1], 
                              self.robot_vel[:, 1]], dim=1) # [num_envs, 4]
        return states, u

    def get_motion_cost(self, state, u, t):
        # State: for each environment, the current state containing position and velocity
        # Action: same but for control input
        
        if 'past_u' not in locals():
            past_u = torch.zeros_like(u, device=self.device)

        # Collision cost via contact forces
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        net_cf = gymtorch.wrap_tensor(_net_cf)
        _net_cf = self.gym.refresh_net_contact_force_tensor(self.sim)
        # Take only forces in x,y in modulus for each environment. Avoid all collisions
        net_cf = torch.sum(torch.abs(torch.cat((net_cf[:, 0].unsqueeze(1), net_cf[:, 1].unsqueeze(1)), 1)),1)
        # The last actors are allowed to collide with eachother (movabable obstacles and robot), check depending on the amount of actors
        allow_dyn_obs = True
        if self.env_type == 'normal':   
            obst_up_to = 6
        elif self.env_type == 'lab':
            obst_up_to = 4 
        elif self.env_type == 'cube':
            obst_up_to = 5
            allow_dyn_obs = False
        
        coll_cost = torch.sum(net_cf.reshape([self.num_envs, int(net_cf.size(dim=0)/self.num_envs)])[:,0:obst_up_to], 1)
        # coll_cost = torch.sum(net_cf.reshape([self.num_envs, int(net_cf.size(dim=0)/self.num_envs)])[:,:], 1) # avoid all obstacles, for navigation
        w_c = 1000 # Weight for collisions
        # Binary check for collisions. So far checking all collision with unmovable obstacles. Movable obstacles touching unmovable ones are considered collisions       
        coll_cost[coll_cost>0.1] = 1
        coll_cost[coll_cost<=0.1] = 0

        # Acceleration cost
        acc_cost = 0.0001*torch.linalg.norm(torch.square((u[0:1]-past_u[0:1])/0.05), dim=1)
        past_u = torch.clone(u)

        # Avoid dynamic obstacle
        penalty_factor = 2 # the larger the factor, the more penalty to geting close to the obs
        dyn_obs_cost = self._predict_dyn_obs(penalty_factor, t+1) if allow_dyn_obs else 0

        return w_c*coll_cost + acc_cost + dyn_obs_cost

    @mppi.handle_batch_input
    def _running_cost(self, state, u, t):
        if self.task == 'navigation' or self.task == 'go_recharge':
            task_cost = self.get_navigation_cost()
        elif self.task == 'push':
            task_cost = self.get_push_cost(False)
        elif self.task == 'pull':
            task_cost = 10 * self.get_pull_cost(False)
        elif self.task == 'push_not_goal':
            task_cost = self.get_push_not_goal_cost()
        elif self.task == 'hybrid':
            task_cost = torch.cat((self.get_push_cost(True)[:int(self.num_envs/2)], self.get_pull_cost(True)[int(self.num_envs/2):]), dim=0)
            # print('push cost', task_cost[:10])
            # print('pull cost', task_cost[self.num_envs-10:])
        elif self.task == 'pick':
            return self.get_panda_pick_cost(True)
        elif self.task == 'place':
            return self.get_panda_place_cost()
        else:
            task_cost = 0

        total_cost = task_cost + self.get_motion_cost(state, u, t)
        
        return  total_cost
    
    # Random test collision avoidance, will be deleted later
    # def get_motion_cost(self, state, u, t):
    #     # State: for each environment, the current state containing position and velocity
    #     # Action: same but for control input
        
    #     if 'past_u' not in locals():
    #         past_u = torch.zeros_like(u, device=self.device)

    #     # Collision cost via contact forces
    #     _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
    #     net_cf = gymtorch.wrap_tensor(_net_cf)
    #     self.gym.refresh_net_contact_force_tensor(self.sim)
    #     # Take only forces in x,y in modulus for each environment. Avoid all collisions
    #     # net_cf = torch.sum(torch.abs(torch.cat((net_cf[:, 0].unsqueeze(1), net_cf[:, 1].unsqueeze(1)), 1)),1)
    #     # The last actors are allowed to collide with eachother (movabable obstacles and robot), check depending on the amount of actors
    #     allow_dyn_obs = True
    #     if self.env_type == 'normal':   
    #         obst_up_to = 6
    #     elif self.env_type == 'lab':
    #         obst_up_to = 4 
    #     elif self.env_type == 'cube':
    #         obst_up_to = 5
    #         allow_dyn_obs = False
        
    #     x_y = net_cf.reshape([self.num_envs, self.bodies_per_env, 3])[:,10, :3]
    #     coll_cost = torch.sum(torch.abs(x_y))
    #     # coll_cost = torch.sum(net_cf.reshape([self.num_envs, int(net_cf.size(dim=0)/self.num_envs)])[:,0:obst_up_to], 1)
    #     # coll_cost = torch.sum(net_cf.reshape([self.num_envs, int(net_cf.size(dim=0)/self.num_envs)])[:,:], 1) # avoid all obstacles, for navigation
    #     w_c = 10 # Weight for collisions
    #     # Binary check for collisions. So far checking all collision with unmovable obstacles. Movable obstacles touching unmovable ones are considered collisions       
    #     # print(torch.max(coll_cost))
    #     # if torch.max(coll_cost) > 50:
    #     #     print('jjjj')
    #     coll_cost[coll_cost>50] = 1
    #     coll_cost[coll_cost<=50] = 0
    #     # coll_cost = 0
    #     # print(coll_cost)

    #     # Acceleration cost
    #     acc_cost = 0.0001*torch.linalg.norm(torch.square((u[0:1]-past_u[0:1])/0.05), dim=1)
    #     past_u = torch.clone(u)

    #     # Avoid dynamic obstacle
    #     penalty_factor = 2 # the larger the factor, the more penalty to geting close to the obs
    #     dyn_obs_cost = self._predict_dyn_obs(penalty_factor, t+1) if allow_dyn_obs else 0

    #     return w_c*coll_cost + acc_cost + dyn_obs_cost