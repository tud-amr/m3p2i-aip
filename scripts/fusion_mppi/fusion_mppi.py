import fusion_mppi.mppi as mppi
import torch, math
from isaacgym import gymtorch, gymapi, torch_utils
from utils import sim_init, skill_utils
import numpy as np

class FUSION_MPPI(mppi.MPPI):
    def __init__(self, dynamics, running_cost, nx, noise_sigma, num_samples=100, horizon=15, device="cpu", 
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
                    env_type="arena",
                    bodies_per_env=None,
                    filter_u=True):
        super().__init__(dynamics, running_cost, nx, noise_sigma, num_samples, horizon, device, 
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

        self.block_goal = torch.tensor([1.5, 3, 0.6], device=self.device)
        self.cube_target_state = None
        
        # Additional variables for the environment or robot
        if self.env_type == "arena":
            self.block_index = 7   # Pushing purple blox, index according to simulation
        if self.env_type == "lab":
            self.block_index = 4  
        if self.env_type == 'store':
            self.block_indexes = np.zeros(self.num_envs)
            self.ee_indexes = np.zeros(self.num_envs)
            if robot_type == 'panda':
                self.block_index = 2
                self.ee_index = 11
                self.block_goal = torch.tensor([0.5, 0, 0.8], device=self.device)
            elif robot_type == 'omni_panda':
                self.block_index = 2
                self.ee_index = 15
                self.block_goal = torch.tensor([1.5, 3, 0.5], device=self.device)
            for i in range(self.num_envs):
                self.block_indexes[i] = self.block_index + i*self.bodies_per_env
                self.ee_indexes[i] = self.ee_index + i*self.bodies_per_env
        if self.env_type == 'shadow':
            self.cube_indexes = np.zeros(self.num_envs)
            self.cube_target_indexes = np.zeros(self.num_envs)
            self.palm_indexes = np.zeros(self.num_envs)
            self.thumb_indexes = np.zeros(self.num_envs)
            self.cube_index = 0
            self.cube_target_index = 1
            self.palm_index = 11
            self.thumb_index = 27
            for i in range(self.num_envs):
                self.cube_indexes[i] = self.cube_index + i*self.bodies_per_env
                self.cube_target_indexes[i] = self.cube_target_index + i*self.bodies_per_env
                self.palm_indexes[i] = self.palm_index + i*self.bodies_per_env
                self.thumb_indexes[i] = self.thumb_index + i*self.bodies_per_env

        self.block_not_goal = torch.tensor([-2, 1], device=self.device)
        self.nav_goal = torch.tensor([3, 3], device=self.device)
        self.panda_hand_goal = torch.tensor([0.5, 0, 0.7, 1, 0, 0, 0], device=self.device)

    def update_gym(self, gym, sim, viewer=None):
        self.gym = gym
        self.sim = sim
        self.viewer = viewer

    def get_navigation_cost(self, r_pos):
        return torch.clamp(torch.linalg.norm(r_pos - self.nav_goal, axis=1)-0.05, min=0, max=1999) 

    def get_push_cost(self, r_pos):
        block_pos = torch.cat((torch.split(torch.clone(self.root_positions[:,0:2]), int(torch.clone(self.root_positions[:,0:2]).size(dim=0)/self.num_envs))),1)[self.block_index,:].reshape(self.num_envs,2)

        robot_to_block = r_pos - block_pos
        block_to_goal = self.block_goal[0:2] - block_pos

        robot_to_block_dist = torch.linalg.norm(robot_to_block, axis = 1)
        block_to_goal_dist = torch.linalg.norm(block_to_goal, axis = 1)
        dist_cost = robot_to_block_dist + block_to_goal_dist 

        # Force the robot behind block and goal,
        # align_cost is actually the cos(theta)

        # Tuning per robot
        if self.robot == "heijn":
            align_weight = 1
            align_offset = 0.1
        elif self.robot == "point_robot":
            align_weight = 0.5
            align_offset = 0.05
        elif self.robot == "boxer":
            align_weight = 1

        align_cost = torch.sum(robot_to_block*block_to_goal, 1)/(robot_to_block_dist*block_to_goal_dist)
        align_cost = align_weight*align_cost
        
        if self.robot != 'boxer':
            align_cost += torch.abs(torch.linalg.norm(r_pos- self.block_goal[:2], axis = 1) - (torch.linalg.norm(block_pos - self.block_goal[:2], axis = 1) + align_offset))

        cost = dist_cost + align_cost

        return cost
    
    def get_pull_cost(self, r_pos):
        block_pos = torch.cat((torch.split(torch.clone(self.root_positions[:,0:2]), int(torch.clone(self.root_positions[:,0:2]).size(dim=0)/self.num_envs))),1)[self.block_index,:].reshape(self.num_envs,2)
        robot_to_block = r_pos - block_pos
        block_to_goal = self.block_goal - block_pos

        robot_to_block_dist = torch.linalg.norm(robot_to_block, axis = 1)
        block_to_goal_dist = torch.linalg.norm(block_to_goal, axis = 1)
        dist_cost = robot_to_block_dist + block_to_goal_dist 

        # Force the robot to be in the middle between block and goal,
        # align_cost is actually the cos(theta)
        align_cost = torch.sum(robot_to_block*block_to_goal, 1)/(robot_to_block_dist*block_to_goal_dist)
        align_cost = (1-align_cost) * 5

        cost = dist_cost + align_cost
        return cost

    def get_push_not_goal_cost(self, r_pos):
        block_pos = torch.cat((torch.split(torch.clone(self.root_positions[:,0:2]), int(torch.clone(self.root_positions[:,0:2]).size(dim=0)/self.num_envs))),1)[self.block_index,:].reshape(self.num_envs,2)
        non_goal_cost = torch.clamp((1/torch.linalg.norm(self.block_not_goal - block_pos,axis = 1)), min=0, max=10)
        return torch.linalg.norm(r_pos - block_pos, axis = 1) + non_goal_cost

    def get_panda_cost(self, joint_pos):
        block_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[self.block_indexes, 0:7]
        self.ee_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[self.ee_indexes, 0:7]
        reach_cost = torch.linalg.norm(self.ee_state[:,0:3] - block_state[:,0:3], axis = 1) 
        goal_cost = torch.linalg.norm(self.block_goal[0:3] - block_state[:,0:3], axis = 1) #+ 2*torch.abs(self.block_goal[2] - block_state[:,2])
        # reach_cost[reach_cost<0.05] = 0*reach_cost[reach_cost<0.05]

        #ee_roll, ee_pitch, _ = torch_utils.get_euler_xyz(self.ee_state[:,3:7])
        
        #align_cost = torch.abs(ee_roll) + torch.abs(ee_pitch)
        return  reach_cost + goal_cost #+ align_cost # 

    def get_shadow_cost(self):
        cube_pos = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[self.cube_indexes, 0:7]
        self.cube_target_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[self.cube_target_indexes, 0:7]
        # self.cube_target_state[:,2] = 0.9
        ort_cost = torch.linalg.norm(cube_pos[:,3:7]-self.cube_target_state[:,3:7], axis = 1)
        pos_cost = torch.linalg.norm(cube_pos[:,:2]-self.cube_target_state[:,:2],  axis = 1)
        #  add fingers in contact
        return 100*pos_cost + ort_cost

    def get_shadow_cost_2(self):
        cube_pos = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[self.cube_indexes, 0:7]
        cube_vel = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[self.cube_indexes, 7:10]
        self.cube_target_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[self.cube_target_indexes, 0:7]
        palm_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[self.palm_indexes, 0:3]
        thumb_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[self.thumb_indexes, 0:3]

        # self.cube_target_state[:,2] = 0.9
        ort_cost = torch.linalg.norm(cube_pos[:,3:7]-self.cube_target_state[:,3:7], axis = 1)
        pos_cost = torch.linalg.norm(cube_pos[:,:3]-palm_state,  axis = 1)
        vel_cost = torch.linalg.norm(cube_vel)
        #  add fingers in contact
        return 40*pos_cost + 3*ort_cost #+  10*vel_cost

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
        obs_lb = torch.tensor([-2.5, 1.5], dtype=torch.float32, device=self.device)
        obs_ub = torch.tensor([-1.5, 2.5], dtype=torch.float32, device=self.device)
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
        actor_root_state = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        dof_states, _, _, _ = sim_init.acquire_states(self.gym, self.sim, print_flag=False)
        self.root_positions = actor_root_state[:, 0:3]
        
        if self.suction_active:
            dof_pos = torch.clone(dof_states).view(-1, 4)[:,[0,2]]
            dof_vel = torch.clone(dof_states).view(-1, 4)[:,[1,3]]
            
            root_pos = torch.reshape(self.root_positions[:, 0:2], (self.num_envs, self.bodies_per_env-2, 2))
            pos_dir = root_pos[:, self.block_index, :] - dof_pos
            # True means the velocity moves towards block, otherwise means pull direction
            flag_towards_block = torch.sum(dof_vel*pos_dir, 1) > 0

            # simulation of a magnetic/suction effect to attach to the box
            suction_force, dir, mask = skill_utils.calculate_suction(root_pos[:, self.block_index, :], dof_pos, self.num_envs, self.kp_suction, self.block_index, self.bodies_per_env)
            # Set no suction force if robot moves towards the block
            suction_force[flag_towards_block] = 0
            # Apply suction/magnetic force
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(torch.reshape(suction_force, (self.num_envs*self.bodies_per_env, 3))), None, gymapi.ENV_SPACE)
            
            # Modify allowed velocities if suction is active, just to have a cone backwards
            rnd_theta = (torch.rand(self.num_envs)*2-1)*120*math.pi/180       # Random angles in a cone of 120 deg
            rot_mat = torch.zeros(self.num_envs, 2, 2, device=self.device)
            rnd_mag = torch.tensor(torch.rand(self.num_envs), device=self.device).reshape([self.num_envs, 1])*2

            # Populate rot matrix
            rot_mat[:, 0, 0] = torch.cos(rnd_theta)
            rot_mat[:, 1, 1] = rot_mat[:, 0, 0]
            rot_mat[:, 0, 1] = -torch.sin(rnd_theta)
            rot_mat[:, 1, 0] = -rot_mat[:, 0, 1]

            dir = dir.reshape([self.num_envs,1,2])
            rnd_input = torch.bmm(dir, rot_mat).squeeze(1)*rnd_mag
            # This is to quickly use the "sample null action" which is the last sample. Should be made more general, considering the N last inputs being priors
            old_last_u = torch.clone(u[-1,:])
            u[mask,:] = rnd_input[mask,:]
            u[-1,:] = old_last_u

        # Use inverse kinematics if the MPPI action space is different than dof velocity space
        u_ = self._ik(u)
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(u_))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        if self.robot == 'boxer':
            res_ = actor_root_state[self.actors_per_env-1::self.actors_per_env]
            res = torch.cat([res_[:, 0:2], res_[:, 7:9]], axis=1)
        elif self.robot == 'point_robot':
            res = torch.clone(dof_states).view(-1, 4)  
        elif self.robot == 'heijn':
            res = torch.clone(dof_states).view(-1, 6)  
        elif self.robot == 'panda':
            res = torch.clone(dof_states).view(-1, 18)
        elif self.robot == 'omni_panda':
            res = torch.clone(dof_states).view(-1, 24)
        elif self.robot == 'shadow_hand':
            res = torch.clone(dof_states).view(-1, 48)

        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

        return res, u


    @mppi.handle_batch_input
    def _running_cost(self, state, u):
        # State: for each environment, the current state containing position and velocity
        # Action: same but for control input
        
        if 'past_u' not in locals():
            past_u = torch.zeros_like(u, device=self.device)

        if self.robot == 'panda':
            state_pos = torch.cat((state[:, 0].unsqueeze(1), state[:, 2].unsqueeze(1), state[:, 4].unsqueeze(1),
                                   state[:, 6].unsqueeze(1), state[:, 8].unsqueeze(1), state[:, 10].unsqueeze(1),
                                   state[:, 12].unsqueeze(1), state[:, 14].unsqueeze(1), state[:, 16].unsqueeze(1)), 1)
        elif self.robot == 'omni_panda':
            state_pos = torch.cat((state[:, 0].unsqueeze(1), state[:, 2].unsqueeze(1), state[:, 4].unsqueeze(1),
                                   state[:, 6].unsqueeze(1), state[:, 8].unsqueeze(1), state[:, 10].unsqueeze(1),
                                   state[:, 12].unsqueeze(1), state[:, 14].unsqueeze(1), state[:, 16].unsqueeze(1),
                                   state[:, 18].unsqueeze(1), state[:, 20].unsqueeze(1), state[:, 22].unsqueeze(1)), 1)   
        # elif self.robot == 'shadow_hand':
        #     state_pos = torch.cat((state[:, 0].unsqueeze(1), state[:, 2].unsqueeze(1), state[:, 4].unsqueeze(1),
        #                            state[:, 6].unsqueeze(1), state[:, 8].unsqueeze(1), state[:, 10].unsqueeze(1),
        #                            state[:, 12].unsqueeze(1), state[:, 14].unsqueeze(1), state[:, 16].unsqueeze(1),
        #                            state[:, 18].unsqueeze(1), state[:, 20].unsqueeze(1), state[:, 22].unsqueeze(1),
        #                            state[:, 24].unsqueeze(1), state[:, 26].unsqueeze(1), state[:, 28].unsqueeze(1),
        #                            state[:, 30].unsqueeze(1), state[:, 32].unsqueeze(1), state[:, 34].unsqueeze(1),
        #                            state[:, 36].unsqueeze(1), state[:, 38].unsqueeze(1), state[:, 40].unsqueeze(1),
        #                            state[:, 42].unsqueeze(1), state[:, 44].unsqueeze(1), state[:, 46].unsqueeze(1)), 1)     
        else:
            state_pos = torch.cat((state[:, 0].unsqueeze(1), state[:, 2].unsqueeze(1)), 1)
        
        control_cost = torch.sum(torch.square(u),1)
        w_u = 0.1
        # Contact forces
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        net_cf = gymtorch.wrap_tensor(_net_cf)
        _net_cf = self.gym.refresh_net_contact_force_tensor(self.sim)
        # Take only forces in x,y in modulus for each environment. Avoid all collisions
        net_cf = torch.sum(torch.abs(torch.cat((net_cf[:, 0].unsqueeze(1), net_cf[:, 1].unsqueeze(1)), 1)),1)
        # The last actors are allowed to collide with eachother (movabable obstacles and robot), check depending on the amount of actors
        if self.env_type == 'arena':   
            obst_up_to = 6
        elif self.env_type == 'lab':
            obst_up_to = 4 
        elif self.env_type == 'store':
            obst_up_to = 2
        elif self.env_type == 'shadow':
            obst_up_to = -1

        if obst_up_to > 0:
            coll_cost = 10000*torch.sum(net_cf.reshape([self.num_envs, int(net_cf.size(dim=0)/self.num_envs)])[:,0:obst_up_to], 1)
        else:
            coll_cost = 0*torch.sum(net_cf.reshape([self.num_envs, int(net_cf.size(dim=0)/self.num_envs)])[:,0:obst_up_to], 1)

        # add collision cost fingers not ro
        # if self.robot == 'panda' or self.robot == 'omni_panda':
        #     gripper_force_cost = torch.sum(0.01*net_cf.reshape([self.num_envs, int(net_cf.size(dim=0)/self.num_envs)])[:,self.bodies_per_env-2:-1],1)
        #     coll_cost += gripper_force_cost
        
        w_c = 1000 # Weight for collisions
        # Binary check for collisions. So far checking all collision with unmovable obstacles. Movable obstacles touching unmovable ones are considered collisions       
        #coll_cost[coll_cost<=0.1] = 0
        
        if self.robot == 'boxer':
            task_cost = self.get_navigation_cost(state[:, :2])
            #task_cost = self.get_push_cost(state[:, :2])
        elif self.robot == 'point_robot':
            #task_cost = self.get_push_cost(state_pos)
            #task_cost = self.get_push_not_goal_cost(state_pos)
            task_cost = self.get_navigation_cost(state_pos)
            #task_cost = self.get_pull_cost(state_pos)
        elif self.robot == 'heijn':
            #task_cost = self.get_navigation_cost(state_pos)
            task_cost = self.get_push_cost(state_pos)
        elif self.robot == 'panda':
            task_cost = self.get_panda_cost(state_pos)
        elif self.robot == 'omni_panda':
            task_cost = self.get_panda_cost(state_pos)
        elif self.robot == 'shadow_hand':
            task_cost = self.get_shadow_cost_2()

        # Acceleration cost
        acc_cost = 0.00001*torch.linalg.norm(torch.square((u[0:1]-past_u[0:1])/0.05), dim=1)
        
        if self.robot == 'panda' or 'omni_panda':
            acc_cost = 0.0001*torch.linalg.norm(torch.square((u-past_u)/0.05), dim=1)
        
        past_u = torch.clone(u)
        
        
        return  task_cost + coll_cost #+ acc_cost # + w_u*control_cost