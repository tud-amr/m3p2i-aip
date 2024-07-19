import torch
from isaacgym import gymtorch, gymapi
from m3p2i_aip.utils import sim_init, skill_utils, mppi_utils
import m3p2i_aip.planners.motion_planner.mppi as mppi

class M3P2I(mppi.MPPI):
    def __init__(self, params, dynamics=None, running_cost=None):
        super().__init__(params, dynamics, running_cost)
        self.kp_suction = 400
        self.suction_active = params.suction_active
        self.env_type = params.environment_type
        
        # Additional variables for the environment
        if self.env_type == "normal":
            self.block_index = 7   # Pushing purple blox, index according to simulation
        if self.env_type == "lab":
            self.block_index = 4  
        self.block_goal = torch.tensor([0, 0], **self.tensor_args)
        self.block_not_goal = torch.tensor([-2, 1], **self.tensor_args)
        self.nav_goal = torch.tensor([3, 3], device=self.device)
        self.task = "navigation"  # "navigation", "push", "pull", "push_not_goal"
        self.align_weight = {"heijn":1, "point_robot":0.5, "boxer":1}
        self.align_offset = {"heijn":0.1, "point_robot":0.05}
        self.goal_quaternion = torch.tensor([0, 0, 0, 1], device=self.device).repeat(self.num_envs).view(self.num_envs, 4)

        # Store obstacle list
        self.allow_dyn_obs = True
        if self.env_type == 'normal':   
            self.obs_list = torch.arange(6, device=self.device)
        elif self.env_type == 'lab':
            self.obs_list = torch.arange(4, device=self.device) 
        elif self.env_type == 'cube':
            self.obs_list = torch.tensor([14, 16], device=self.device) 
            self.allow_dyn_obs = False
        elif self.env_type == 'albert_arena':
            self.obs_list = torch.tensor(0, device=self.device) 
            self.allow_dyn_obs = False
        # self.obs_list = torch.arange(self.bodies_per_env, device=self.device) # avoid all obstacles

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
        # if self.robot == 'albert':
        #     self.cube_goal_state = torch.tensor([0.5, 0.2, 0.7, 0, 0, 0, 1], device='cuda:0')
    
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
        
    def update_infinite_beta(self, costs, beta, eta_u_bound, eta_l_bound):
        """
            Update the inverse temperature on the fly
        """
        found = False
        # Makes sure beta is properly tuned before computing the weights
        while not found:
        # Normalization of the weights
            exp_ = torch.exp((-1.0/beta) * costs)
            eta = torch.sum(exp_)       # tells how many significant samples we have, more or less

            # Update beta to make eta converge within the bounds
            beta_lm = 0.9
            beta_um = 1.2
            if eta > eta_u_bound:
                beta = beta * beta_lm
            elif eta < eta_l_bound:
                beta = beta * beta_um
            else:
                found = True
        return eta, exp_

    def _multi_modal_exp_util(self, costs):
        """
           Calculate weights using exponential utility given cost
           Iuput: costs [K, T], costs within horizon
        """
        traj_costs = mppi_utils.cost_to_go(costs, self.gamma_seq) # [K, T]
        traj_costs = traj_costs[:,0] # [K] Costs for the next timestep

        total_costs_1 = traj_costs[:self.half_K] - torch.min(traj_costs[:self.half_K])
        total_costs_2 = traj_costs[self.half_K:] - torch.min(traj_costs[self.half_K:])
        total_costs = traj_costs - torch.min(traj_costs)
        # print('1', total_costs_1)
        # print('2', total_costs_2)
        eta_1, exp_1 = self.update_infinite_beta(total_costs_1, self.beta_1, 10, 3)
        eta_2, exp_2 = self.update_infinite_beta(total_costs_2, self.beta_2, 10, 3)
        eta, exp_ = self.update_infinite_beta(total_costs, self.beta, 10, 3)
        # exp_ = torch.exp((-1.0/self.beta) * total_costs)
        # eta = torch.sum(exp_)

        self.weights_1 = 1 / eta_1 * exp_1 
        self.weights_2 = 1 / eta_2 * exp_2
        self.weights = 1 / eta * exp_ 
        # print('weights', self.weights.size())
    
    def _update_multi_modal_distribution(self, costs, actions):
        """
            Update moments using sample trajectories.
            So far only mean is updated, eventually one could also update the covariance
        """

        self._multi_modal_exp_util(costs)

        # # Update best action
        self.best_idx_1 = torch.argmax(self.weights_1)
        self.best_idx_2 = torch.argmax(self.weights_2)
        self.best_traj_1 = torch.index_select(actions, 0, self.best_idx_1).squeeze(0)
        self.best_traj_2 = torch.index_select(actions, 0, self.best_idx_2+self.half_K).squeeze(0)
       
        weighted_seq = self.weights.view(-1, 1, 1) * actions # [K, T, nu]
        # print(actions)
        self.mean_action_1 = torch.sum(self.weights_1.view(-1, 1, 1) * actions[:self.half_K], dim=0)
        self.mean_action_2 = torch.sum(self.weights_2.view(-1, 1, 1) * actions[self.half_K:], dim=0)

        # Gradient update for the mean
        self.mean_action = (1.0 - self.step_size_mean) * self.mean_action +\
            self.step_size_mean * torch.sum(weighted_seq, 0)
        # print(torch.sum(weighted_seq, 0))
        # print(self.mean_action.size()) # [T, nu]
       
        delta = actions - self.mean_action.unsqueeze(0)

        return delta

    def get_navigation_cost(self):
        return torch.clamp(torch.linalg.norm(self.robot_pos - self.nav_goal, axis=1)-0.05, min=0, max=1999) 
    
    def calculate_dist(self):
        self.robot_to_block = self.robot_pos - self.block_pos
        self.block_to_goal = self.block_goal - self.block_pos

        self.robot_to_block_dist = torch.linalg.norm(self.robot_to_block, axis = 1)
        self.block_to_goal_dist = torch.linalg.norm(self.block_to_goal, axis = 1)
        
        self.dist_cost = self.robot_to_block_dist + self.block_to_goal_dist * 10
        self.cos_theta = torch.sum(self.robot_to_block*self.block_to_goal, 1)/(self.robot_to_block_dist*self.block_to_goal_dist)

    def get_push_cost(self):
        # Calculate dist cost
        self.calculate_dist()

        # Force the robot behind block and goal, align_cost is actually cos(theta)+1
        # align_cost = self.align_weight[self.robot] * (self.cos_theta + 1) * 5
        align_cost = torch.zeros(self.num_envs, **self.tensor_args)
        align_cost[self.cos_theta>0] = self.cos_theta[self.cos_theta>0]
        # print('push align', align_cost[:10])
        # if self.robot != 'boxer':
        #     align_cost += torch.abs(self.robot_to_goal_dist - self.block_to_goal_dist - self.align_offset[self.robot])
        ori_cost = skill_utils.get_general_ori_cube2goal(self.block_quat, self.goal_quaternion)
 
        return 3 * self.dist_cost + 1 * align_cost #+ 10 * ori_cost# [num_envs] 31
    
    def get_pull_cost(self, hybrid):
        pos_dir = self.block_pos - self.robot_pos
        # True means the velocity moves towards block, otherwise means pull direction
        flag_towards_block = torch.sum(self.robot_vel*pos_dir, 1) > 0

        # simulation of a magnetic/suction effect to attach to the box
        suction_force, dir, mask = skill_utils.calculate_suction(self.block_pos, self.robot_pos, self.num_envs, self.kp_suction, self.block_index, self.bodies_per_env)
        # Set no suction force if robot moves towards the block
        suction_force[flag_towards_block] = 0
        if hybrid:
            suction_force[:self.half_K] = 0
        # Apply suction/magnetic force
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(torch.reshape(suction_force, (self.num_envs*self.bodies_per_env, 3))), None, gymapi.ENV_SPACE)

        # Calculate dist cost
        self.calculate_dist()

        # Force the robot to be in the middle between block and goal, align_cost is actually 1-cos(theta)
        # align_cost = (1 - self.cos_theta) * 5
        align_cost = torch.zeros(self.num_envs, **self.tensor_args)
        align_cost[self.cos_theta<0] = -self.cos_theta[self.cos_theta<0]
        # print('pull align', align_cost[-10:])

        # Add the cost when the robot is close to the block and moves towards the block
        vel_cost = torch.zeros(self.num_envs, **self.tensor_args)
        robot_block_close = self.robot_to_block_dist <= 0.5
        vel_cost[flag_towards_block*robot_block_close] = 0.6

        ori_cost = skill_utils.get_general_ori_cube2goal(self.block_quat, self.goal_quaternion)

        return 3 * self.dist_cost + vel_cost + 5 * align_cost #+ 10 * ori_cost # [num_envs] 315 

    def get_push_not_goal_cost(self):
        non_goal_cost = torch.clamp((1/torch.linalg.norm(self.block_not_goal - self.block_pos,axis = 1)), min=0, max=10)
        return torch.linalg.norm(self.robot_pos - self.block_pos, axis = 1) + non_goal_cost

    def _predict_dyn_obs(self, factor, t):
        # Obs boundary [-2.5, 1.5] <--> [-1.5, 2.5]
        obs_lb = torch.tensor([-2.5, 1.5], **self.tensor_args)
        obs_ub = torch.tensor([-1.5, 2.5], **self.tensor_args)
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
        threshold_gripper = {'panda':0.1, 'albert':0.08}
        gripper_cost[reach_cost < threshold_gripper[self.robot]] = 0

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
        weight_goal = {True:15, False:5}
        total_cost = 0.2 * manip_cost + 10 * reach_cost + weight_goal[self.multi_modal] * goal_cost + ori_cost + gripper_cost + tilt_cost

        # if self.robot == 'albert':
        #     vel_cost = torch.linalg.norm(self.robot_vel, axis=1)
        #     vel_cost[goal_cost > 0.25] = 0
        #     print(vel_cost)
        #     total_cost += 5 * vel_cost
        return  total_cost #+ align_cost multiply 10*reach_cost when using mppi_mode == storm

    def get_pick_tilt_cost(self, hybrid):
        # This measures the cost of the tilt angle between the end effector and the cube
        ee_quaternion = self.ee_l_state[:, 3:7]
        cube_quaternion = self.cube_state[:, 3:7]
        if not hybrid:
            # To make the z-axis direction of end effector to be perpendicular to the cube surface
            ori_ee2cube = skill_utils.get_general_ori_ee2cube(ee_quaternion, cube_quaternion, tilt_value=0)
        else:
            # To combine costs of different tilt angles
            cost_1 = skill_utils.get_general_ori_ee2cube(ee_quaternion[:self.half_K], 
                                                         cube_quaternion[:self.half_K], tilt_value = 0)
            cost_2 = skill_utils.get_general_ori_ee2cube(ee_quaternion[self.half_K:], 
                                                         cube_quaternion[self.half_K:], tilt_value = 0.5) #0.9
            ori_ee2cube =  torch.cat((cost_1, cost_2), dim=0)

        return 3 * ori_ee2cube

    def get_panda_place_cost(self):
        gripper_dist = torch.linalg.norm(self.ee_l_state[:, :3] - self.ee_r_state[:, :3], axis=1)
        gripper_cost = 1 - gripper_dist
        self.ee_state = (self.ee_l_state + self.ee_r_state) / 2
        reach_cost = torch.linalg.norm(self.ee_state[:,:7] - self.ee_goal[:7], axis=1)
        # If gripper is not fully open, no reach cost
        reach_cost[gripper_dist <= 0.078] = 0
        # If gripper is fully open, no gripper cost, retract the arm
        gripper_cost[gripper_dist > 0.078] = 0
        if self.robot == 'albert':
            vel_cost = torch.linalg.norm(self.robot_vel, axis=1)
            return 10 * gripper_cost + 10 * vel_cost
        elif self.robot == 'panda':
            return 10 * gripper_cost + 10 * reach_cost
    
    def get_albert_cost(self):
        # nav_cost = 10 * torch.clamp(torch.linalg.norm(self.robot_pos - self.nav_goal, axis=1)-0.05, min=0, max=1999)
        # # print(cost)
        # base_quaternion = self.robot_states[:,3:7]
        # base_goal_quaternion = torch.tensor([0, 0, 0.714, 0.714], device='cuda:0').repeat(self.num_envs).view(self.num_envs, 4)
        # nav_ori = 3 * skill_utils.get_ori_cube2goal(base_quaternion, base_goal_quaternion)
        # nav_cost += nav_ori
        # print(self.robot_states)
        pick_cost = self.get_panda_pick_cost(False)

        gripper_dist = torch.linalg.norm(self.ee_l_state[:, :3] - self.ee_r_state[:, :3], axis=1)
        gripper_cost = 1 - gripper_dist
        goal_cost = torch.linalg.norm(self.cube_goal_state[:3] - self.cube_state[:,:3], axis = 1)
        gripper_cost[goal_cost>0.1]=0
        pick_cost[goal_cost<0.1] = 0
        return pick_cost #+ 100 * gripper_cost

    @mppi.handle_batch_input
    def _dynamics(self, state, u, t):
        # Use inverse kinematics if the MPPI action space is different than dof velocity space
        u_ = skill_utils.apply_ik(self.robot, u) # forward simulate for the rollouts
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(u_))
        
        # Step the simulation
        sim_init.step(self.gym, self.sim)  # very essential 0.002s
        sim_init.refresh_states(self.gym, self.sim)
        sim_init.step_rendering(self.gym, self.sim, self.viewer, sync_frame_time=True)

        # Return the current states
        states = torch.stack([self.robot_pos[:, 0], 
                              self.robot_vel[:, 0], 
                              self.robot_pos[:, 1], 
                              self.robot_vel[:, 1]], dim=1) # [num_envs, 4]
        return states, u

    def get_motion_cost(self, t):
        # Collision cost via contact forces
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        net_cf = gymtorch.wrap_tensor(_net_cf) # [total_num_bodies, 3]
        _net_cf = self.gym.refresh_net_contact_force_tensor(self.sim)

        # Take only forces in x,y in modulus for each environment.
        net_cf_xy = torch.sum(torch.abs(net_cf[:, :2]),1) # [total_num_bodies]
        net_cf_xy = net_cf_xy.reshape([self.num_envs, self.bodies_per_env])

        # Consider collision costs from obstacle list
        coll_cost = torch.sum(torch.index_select(net_cf_xy, 1, self.obs_list), 1) # [num_envs]
        w_c = 1000
        # Binary check for collisions.
        coll_cost[coll_cost>0.1] = 1
        coll_cost[coll_cost<=0.1] = 0

        # Avoid dynamic obstacle
        penalty_factor = 2 # the larger the factor, the more penalty to geting close to the obs
        dyn_obs_cost = self._predict_dyn_obs(penalty_factor, t+1) if self.allow_dyn_obs else 0

        return w_c*coll_cost + dyn_obs_cost

    @mppi.handle_batch_input
    def _running_cost(self, state, u, t):
        # if self.robot == 'albert':
        #     return self.get_albert_cost()
        if self.task == 'navigation' or self.task == 'go_recharge':
            task_cost = self.get_navigation_cost()
        elif self.task == 'push':
            task_cost = self.get_push_cost()
        elif self.task == 'pull':
            task_cost = self.get_pull_cost(False) # 10
        elif self.task == 'push_not_goal':
            task_cost = self.get_push_not_goal_cost()
        elif self.task == 'hybrid':
            return torch.cat((self.get_push_cost()[:self.half_K], self.get_pull_cost(True)[self.half_K:]), dim=0)
            # print('push cost', task_cost[:10])
            # print('pull cost', task_cost[self.num_envs-10:])
        elif self.task == 'pick':
            # return self.get_panda_pick_cost(self.multi_modal) # for albert
            task_cost = self.get_panda_pick_cost(self.multi_modal) # for panda
        elif self.task == 'place':
            return self.get_panda_place_cost()
        else:
            task_cost = 0

        total_cost = task_cost + self.get_motion_cost(t)
        
        return  total_cost