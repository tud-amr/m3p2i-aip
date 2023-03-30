import fusion_mppi.mppi as mppi
import torch, math
import time
from isaacgym import gymtorch, gymapi
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
            self.robot_pos = states_dict["robot_pos"]
            self.robot_vel = states_dict["robot_vel"]
            self.flag = False
    
    def update_task(self, task, goal):
        self.task = task
        if self.task in ['navigation', 'go_recharge']:
            self.nav_goal = goal
        elif self.task in ['push', 'pull', 'hybrid']:
            self.block_goal = goal
    
    def update_params(self, params):
        self.params = params
        if self.task == 'hybrid':
            self.params.suction_active = True
        else:
            self.suction_active = self.params.suction_active

    def get_navigation_cost(self):
        return torch.clamp(torch.linalg.norm(self.robot_pos - self.nav_goal, axis=1)-0.05, min=0, max=1999) 

    def get_push_cost(self):
        robot_to_block = self.robot_pos - self.block_pos
        block_to_goal = self.block_goal - self.block_pos

        robot_to_block_dist = torch.linalg.norm(robot_to_block, axis = 1)
        block_to_goal_dist = torch.linalg.norm(block_to_goal, axis = 1)
        # robot_to_goal_dist = torch.linalg.norm(self.robot_pos- self.block_goal, axis = 1)
        dist_cost = robot_to_block_dist + block_to_goal_dist 

        # Force the robot behind block and goal,
        # align_cost is actually the cos(theta)+1
        align_cost = torch.sum(robot_to_block*block_to_goal, 1)/(robot_to_block_dist*block_to_goal_dist)
        align_cost = self.align_weight[self.robot] * (align_cost+1) * 5
        # if self.robot != 'boxer':
        #     align_cost += torch.abs(robot_to_goal_dist - block_to_goal_dist - self.align_offset[self.robot])
        cost = dist_cost + align_cost

        return cost
    
    def get_pull_cost(self):
        pos_dir = self.block_pos - self.robot_pos
        # True means the velocity moves towards block, otherwise means pull direction
        flag_towards_block = torch.sum(self.robot_vel*pos_dir, 1) > 0

        # simulation of a magnetic/suction effect to attach to the box
        suction_force, dir, mask = skill_utils.calculate_suction(self.block_pos, self.robot_pos, self.num_envs, self.kp_suction, self.block_index, self.bodies_per_env)
        # Set no suction force if robot moves towards the block
        suction_force[flag_towards_block] = 0
        # Apply suction/magnetic force
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(torch.reshape(suction_force, (self.num_envs*self.bodies_per_env, 3))), None, gymapi.ENV_SPACE)

        robot_to_block = self.robot_pos - self.block_pos
        block_to_goal = self.block_goal - self.block_pos

        robot_to_block_dist = torch.linalg.norm(robot_to_block, axis = 1)
        block_to_goal_dist = torch.linalg.norm(block_to_goal, axis = 1)
        dist_cost = robot_to_block_dist + block_to_goal_dist # * 5

        # Force the robot to be in the middle between block and goal,
        # align_cost is actually the 1-cos(theta)
        align_cost = torch.sum(robot_to_block*block_to_goal, 1)/(robot_to_block_dist*block_to_goal_dist)
        align_cost = (1-align_cost) * 5

        cost = dist_cost + align_cost # [num_envs]
        return cost

    def get_push_not_goal_cost(self):
        non_goal_cost = torch.clamp((1/torch.linalg.norm(self.block_not_goal - self.block_pos,axis = 1)), min=0, max=10)
        return torch.linalg.norm(self.robot_pos - self.block_pos, axis = 1) + non_goal_cost

    @mppi.handle_batch_input
    def _ik(self, u):
        if self.robot == 'boxer':
            r = 0.08
            L = 2*0.157
            # Diff drive fk
            u_ik = u.clone()
            u_ik[:, 0] = (u[:, 0] / r) - ((L*u[:, 1])/(2*r))
            u_ik[:, 1] = (u[:, 0] / r) + ((L*u[:, 1])/(2*r))
            return u_ik
        else: return u

    def _predict_dyn_obs(self, factor, robot_state, dyn_obs_pos, dyn_obs_vel, t):
        robot_pos = robot_state[:, [0, 2]] # K x 2
        # Obs boundary [-2.5, 1.5] <--> [-1.5, 2.5]
        obs_lb = torch.tensor([-2.5, 1.5], dtype=torch.float32, device="cuda:0")
        obs_ub = torch.tensor([-1.5, 2.5], dtype=torch.float32, device="cuda:0")
        dyn_obs_vel = torch.clamp(dyn_obs_vel, min = -0.001, max = 0.001)
        pred_pos = dyn_obs_pos + t * dyn_obs_vel * 10
        # Check the prec_pos and boundary
        exceed_ub = pred_pos[:, 1] > obs_ub[1]
        exceed_lb = pred_pos[:, 1] < obs_lb[1]
        pred_pos[exceed_ub] = 2 * obs_ub - pred_pos[exceed_ub]
        pred_pos[exceed_lb] = 2 * obs_lb - pred_pos[exceed_lb]
        # Compute the cost
        dyn_obs_cost = factor * torch.exp(-torch.norm(pred_pos - robot_pos, dim=1))

        return dyn_obs_cost

    @mppi.handle_batch_input
    def _dynamics(self, state, u, t):
        # Apply suction force
        # if self.suction_active:
            # pos_dir = self.block_pos - self.robot_pos
            # # True means the velocity moves towards block, otherwise means pull direction
            # flag_towards_block = torch.sum(self.robot_vel*pos_dir, 1) > 0

            # # simulation of a magnetic/suction effect to attach to the box
            # suction_force, dir, mask = skill_utils.calculate_suction(self.block_pos, self.robot_pos, self.num_envs, self.kp_suction, self.block_index, self.bodies_per_env)
            # # Set no suction force if robot moves towards the block
            # suction_force[flag_towards_block] = 0
            # # Apply suction/magnetic force
            # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(torch.reshape(suction_force, (self.num_envs*self.bodies_per_env, 3))), None, gymapi.ENV_SPACE)
            
            # # Modify allowed velocities if suction is active, just to have a cone backwards
            # rnd_theta = (torch.rand(self.num_envs)*2-1)*120*math.pi/180       # Random angles in a cone of 120 deg
            # rot_mat = torch.zeros(self.num_envs, 2, 2, device="cuda:0")
            # rnd_mag = torch.tensor(torch.rand(self.num_envs), device='cuda:0').reshape([self.num_envs, 1])*2

            # # Populate rot matrix
            # rot_mat[:, 0, 0] = torch.cos(rnd_theta)
            # rot_mat[:, 1, 1] = rot_mat[:, 0, 0]
            # rot_mat[:, 0, 1] = -torch.sin(rnd_theta)
            # rot_mat[:, 1, 0] = -rot_mat[:, 0, 1]

            # dir = dir.reshape([self.num_envs,1,2])
            # rnd_input = torch.bmm(dir, rot_mat).squeeze(1)*rnd_mag
            # # print('u', u.size()) # [200, 3] for heijn
            # # print('rnd input', rnd_input.size()) # [200, 2]
            # # This is to quickly use the "sample null action" which is the last sample. Should be made more general, considering the N last inputs being priors
            # old_last_u = torch.clone(u[-1,:])
            # u[mask,:2] = rnd_input[mask,:]
            # u[-1,:] = old_last_u

        # Use inverse kinematics if the MPPI action space is different than dof velocity space
        u_ = self._ik(u)
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

    def get_motion_cost(self, state, u):
        # State: for each environment, the current state containing position and velocity
        # Action: same but for control input
        
        if 'past_u' not in locals():
            past_u = torch.zeros_like(u, device=self.device)

        # state_pos = torch.cat((state[:, 0].unsqueeze(1), state[:, 2].unsqueeze(1)), 1)
        # control_cost = torch.sum(torch.square(u),1)
        # w_u = 0.1

        # Collision cost via contact forces
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        net_cf = gymtorch.wrap_tensor(_net_cf)
        _net_cf = self.gym.refresh_net_contact_force_tensor(self.sim)
        # Take only forces in x,y in modulus for each environment. Avoid all collisions
        net_cf = torch.sum(torch.abs(torch.cat((net_cf[:, 0].unsqueeze(1), net_cf[:, 1].unsqueeze(1)), 1)),1)
        # The last actors are allowed to collide with eachother (movabable obstacles and robot), check depending on the amount of actors
        if self.env_type == 'normal':   
            obst_up_to = 6
        elif self.env_type == 'lab':
            obst_up_to = 4 
        
        coll_cost = torch.sum(net_cf.reshape([self.num_envs, int(net_cf.size(dim=0)/self.num_envs)])[:,0:obst_up_to], 1)
        # coll_cost = torch.sum(net_cf.reshape([self.num_envs, int(net_cf.size(dim=0)/self.num_envs)])[:,:], 1) # avoid all obstacles, for navigation
        w_c = 1000 # Weight for collisions
        # Binary check for collisions. So far checking all collision with unmovable obstacles. Movable obstacles touching unmovable ones are considered collisions       
        coll_cost[coll_cost>0.1] = 1
        coll_cost[coll_cost<=0.1] = 0

        # Acceleration cost
        acc_cost = 0.0001*torch.linalg.norm(torch.square((u[0:1]-past_u[0:1])/0.05), dim=1)
        past_u = torch.clone(u)

        return w_c*coll_cost + acc_cost # + w_u*control_cost

    @mppi.handle_batch_input
    def _running_cost(self, state, u):
        if self.task == 'navigation' or self.task == 'go_recharge':
            task_cost = self.get_navigation_cost()
        elif self.task == 'push':
            task_cost = self.get_push_cost()
        elif self.task == 'pull':
            task_cost = self.get_pull_cost()
        elif self.task == 'push_not_goal':
            task_cost = self.get_push_not_goal_cost()
        elif self.task == 'hybrid':
            task_cost = torch.cat((self.get_push_cost()[:int(self.num_envs/2)], self.get_pull_cost()[int(self.num_envs/2):]), dim=0)

        total_cost = task_cost + self.get_motion_cost(state, u)
        
        return  total_cost