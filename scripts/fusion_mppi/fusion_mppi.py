import fusion_mppi.mppi as mppi
import torch, math
from isaacgym import gymtorch, gymapi, torch_utils
from utils import sim_init, skill_utils
import numpy as np
import pytorch3d.transforms

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
        self.ort_goal_euler = torch.tensor([0, 0, 0], device=self.device)

        self.block_goal = torch.tensor([0.4, 0, 0.6], device=self.device)
        self.cube_target_state = None

        # Comparison with baselines
        # -----------------------------------------------------------------------------------------------------------------------------------
        self.block_goal_pose_emdn_1 = torch.tensor([0.5, 0.3, 0.5, 0.0, 0.0, 0.0, 1.0], device=self.device)
        self.block_goal_pose_emdn_2 = torch.tensor([0.5, 0.3, 0.5, 0, 0, 0.7071068, 0.7071068], device=self.device) # Rotation 90 deg

        self.block_goal_pose_ur5_c = torch.tensor([0.7, -0.05, 0.5, 0, 0, 0, 1], device=self.device)
        self.block_goal_pose_ur5_l= torch.tensor([0.7, 0.2, 0.5,  0, 0, 0.258819, 0.9659258 ], device=self.device) # Rotation 30 deg
        self.block_goal_pose_ur5_r= torch.tensor([0.7, -0.2, 0.5,  0, 0, -0.258819, 0.9659258 ], device=self.device) # Rotation -30 deg

        # Select goal according to test
        self.block_goal_pose = torch.clone(self.block_goal_pose_ur5_r)
        # -----------------------------------------------------------------------------------------------------------------------------------

        self.block_goal_ort = torch.tensor([0.0, 0.0, 0.0, 1], device=self.device)
        # Counter for periodic pinting
        self.count = 0

        # Additional variables for the environment or robot
        if self.env_type == "arena":
            self.block_index = 7   # Pushing purple blox, index according to simulation
            self.block_goal = torch.tensor([3, -3, 0.6], device=self.device)
        if self.env_type == "lab":
            self.block_index = 4  
        if self.env_type == 'store':
            self.block_indexes = np.zeros(self.num_envs)
            self.ee_indexes = np.zeros(self.num_envs)
            if robot_type == 'panda':
                self.block_index = 2
                self.ee_index = 12
            elif robot_type == 'omni_panda':
                self.block_index = 2
                self.ee_index = 15
                self.block_goal = torch.tensor([1.5, 3., 0.6], device=self.device)
            elif robot_type == 'albert':
                self.block_index = 2
                self.ee_index = 21
                self.ee_goal = torch.tensor([1.5, 3., 0.6], device=self.device)
            elif robot_type == 'panda_no_hand':
                self.block_index = 2
                self.ee_index = 12
                self.block_goal = torch.tensor([0.5, 0.0, 0.138], device=self.device)
            for i in range(self.num_envs):
                self.block_indexes[i] = self.block_index + i*self.bodies_per_env
                self.ee_indexes[i] = self.ee_index + i*self.bodies_per_env
        elif self.env_type == 'lab':
            self.block_goal = torch.tensor([0., 0, 0.],device=self.device)
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
        if self.env_type == 'storm':
            self.ee_indexes = np.zeros(self.num_envs)
            self.goal_indexes = np.zeros(self.num_envs)
            self.ee_index = 12
            self.goal_index = 3
            for i in range(self.num_envs):
                self.ee_indexes[i] = self.ee_index + i*self.bodies_per_env
                self.goal_indexes[i] = self.goal_index + i*self.bodies_per_env
        
        
        self.block_not_goal = torch.tensor([-2, 1], device=self.device)
        self.nav_goal = torch.tensor([3, 3], device=self.device)
        self.panda_hand_goal = torch.tensor([0.5, 0.0, 0.7, 1, 0, 0, 0], device=self.device)
        self.joint_comfy = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.58, 0., 1.86, 0.], device=self.device)
    def update_gym(self, gym, sim, viewer=None):
        self.gym = gym
        self.sim = sim
        self.viewer = viewer

    def get_navigation_cost(self, r_pos):
        return torch.clamp(torch.linalg.norm(r_pos - self.nav_goal, axis=1)-0.05, min=0, max=1999) 
    
    def get_albert_cost(self, r_pos):
        self.ee_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[self.ee_indexes, 0:7]
        reach_cost = torch.linalg.norm(self.ee_state[:,0:3] - self.ee_goal, axis = 1) 
        return reach_cost 

    def get_push_cost(self, r_pos):
        block_pos = torch.cat((torch.split(torch.clone(self.root_positions[:,0:2]), int(torch.clone(self.root_positions[:,0:2]).size(dim=0)/self.num_envs))),1)[self.block_index,:].reshape(self.num_envs,2)
        block_ort = torch.cat((torch.split(torch.clone(self.root_ort), int(torch.clone(self.root_ort).size(dim=0)/self.num_envs))),1)[self.block_index,:].reshape(self.num_envs,4)

        robot_to_block = r_pos - block_pos
        block_to_goal = self.block_goal[0:2] - block_pos
        block_to_goal_ort = self.orientation_error(self.block_goal_ort, block_ort)
        
        robot_to_block_dist = torch.linalg.norm(robot_to_block, axis = 1)
        block_to_goal_dist = torch.linalg.norm(block_to_goal, axis = 1)
        dist_cost = robot_to_block_dist + block_to_goal_dist #+ 1*block_to_goal_ort

        # Force the robot behind block and goal,
        # align_cost is actually the cos(theta)

        # Tuning per robot
        if self.robot == "heijn":
            align_weight = 1
            # align_offset = 0.1
            w_d = 5
        elif self.robot == "point_robot":
            align_weight = 0.5
            # align_offset = 0.05
            w_d = 1
        elif self.robot == "boxer":
            align_weight = 1
            w_d = 1

        align_cost = torch.sum(robot_to_block*block_to_goal, 1)/(robot_to_block_dist*block_to_goal_dist)
        align_cost = align_weight*align_cost
        
        # if self.robot != 'boxer':
        #     align_cost += torch.abs(torch.linalg.norm(r_pos- self.block_goal[:2], axis = 1) - (torch.linalg.norm(block_pos - self.block_goal[:2], axis = 1) + align_offset))
        
        # print('dist', dist_cost[-1])
        # print('align', align_cost[-1])
        cost = w_d*dist_cost + 3*align_cost

        return cost

    def orientation_error(self, q1, q2_batch):
        """
        Computes the orientation error between a single quaternion and a batch of quaternions.
        
        Parameters:
        -----------
        q1 : torch.Tensor
            A tensor of shape (4,) representing the first quaternion.
        q2_batch : torch.Tensor
            An tensor of shape (batch_size, 4) representing the second set of quaternions.
            
        Returns:
        --------
        error_batch : torch.Tensor
            An tensor of shape (batch_size,) containing the orientation error between the first quaternion and each quaternion in the batch.
        """
        
        # Expand the first quaternion to match the batch size of the second quaternion
        q1_batch = q1.expand(q2_batch.shape[0], -1)
        
        # Normalize the quaternions
        q1_batch = q1_batch / torch.norm(q1_batch, dim=1, keepdim=True)
        q2_batch = q2_batch / torch.norm(q2_batch, dim=1, keepdim=True)
        
        # Compute the dot product between the quaternions in the batch
        dot_product_batch = torch.sum(q1_batch * q2_batch, dim=1)
        
        # Compute the angle between the quaternions in the batch
        # chatgpt
        #angle_batch = 2 * torch.acos(torch.abs(dot_product_batch))
        # method2
        angle_batch = torch.acos(2*torch.square(dot_product_batch)-1)
        # method 3
        # angle_batch = 1 - torch.square(dot_product_batch)
        # Return the orientation error for each quaternion in the batch
        error_batch = angle_batch
        #print(error_batch[-1])
        return error_batch

    def get_panda_push_cost(self, joint_pos):
        r_pose = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[self.ee_indexes, 0:7]
        r_pos = r_pose[:,0:3]
        r_ort = r_pose[:,3:7]
        ee_height = r_pose[:,2]
        block_pos = torch.cat((torch.split(torch.clone(self.root_positions[:,0:3]), int(torch.clone(self.root_positions[:,0:3]).size(dim=0)/self.num_envs))),1)[self.block_index,:].reshape(self.num_envs,3)
        block_ort = torch.cat((torch.split(torch.clone(self.root_ort), int(torch.clone(self.root_ort).size(dim=0)/self.num_envs))),1)[self.block_index,:].reshape(self.num_envs,4)
        robot_to_block = r_pos - block_pos

        # block_to_goal = self.block_goal[0:2] - block_pos[:,0:2]
        block_to_goal = self.block_goal_pose[0:2] - block_pos[:,0:2]

        block_to_goal_ort = self.orientation_error(self.block_goal_pose[3:7], block_ort)

        robot_to_block_dist = torch.linalg.norm(robot_to_block[:, 0:2], axis = 1)
        block_to_goal_dist = torch.linalg.norm(block_to_goal, axis = 1)

        # print(block_to_goal_ort)
        # block_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(block_ort), "ZYX")

        block_to_goal_ort = torch.nan_to_num(block_to_goal_ort, nan=1.0)

        # block_yaw = torch.atan2(2.0 * (block_ort[:,-1] * block_ort[:,2] + block_ort[:,0] * block_ort[:,1]), block_ort[:,-1] * block_ort[:,-1] + block_ort[:,0] * block_ort[:,0] - block_ort[:,1] * block_ort[:,1] - block_ort[:,2] * block_ort[:,2])

        hoover_height = 0.130
        ee_hover_cost= torch.abs(ee_height - hoover_height) 
        dist_cost = 20*robot_to_block_dist + 100*block_to_goal_dist #+ 10*block_to_goal_ort

        robot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(r_ort), "ZYX")

        ee_align_cost = torch.linalg.norm(robot_euler - self.ort_goal_euler, axis=1)

        align_cost = torch.sum(robot_to_block[:,0:2]*block_to_goal, 1)/(robot_to_block_dist*block_to_goal_dist)
        posture_cost = align_cost + 4*ee_align_cost + 20*ee_hover_cost
        
        # Evaluation metrics 
        if self.count > 300:
            # Comparison
            # Ex = torch.abs(self.block_goal[0]-block_pos[-1,0])
            # Ey = torch.abs(self.block_goal[1]-block_pos[-1,1])
            Ex = torch.abs(self.block_goal_pose[0]-block_pos[-1,0])
            Ey = torch.abs(self.block_goal_pose[1]-block_pos[-1,1])
            Etheta = torch.abs(block_to_goal_ort[-1])
            
            metric_1 = 1.5*(Ex+Ey)+0.01*Etheta
            print("Metric Baxter", metric_1)
            print("Angle", Etheta)
            if Ex < 0.025 and Ey < 0.01 and Etheta < 0.052:
                print("Success")

            self.count = 0
        else:
            self.count +=1

        return dist_cost + 3*posture_cost

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

        ee_roll, ee_pitch, _ = torch_utils.get_euler_xyz(self.ee_state[:,3:7])

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

        align_cost = torch.abs(ee_pitch) + torch.abs(ee_roll-3.14)
        return  0.2*manip_cost + 10*reach_cost + 5*goal_cost # + align_cost multiply 10*reach_cost when using mppi_mode == storm

    def get_panda_reach_cost(self, joint_pos):
        self.ee_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[self.ee_indexes, 0:7]
        goal_pos = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[self.goal_indexes, 0:7]
        goal_pos[:, 3:7] =  self.panda_hand_goal[3:7]
        reach_cost = torch.linalg.norm(self.ee_state[:,:3] - goal_pos[:,:3], axis = 1) 
        align_cost = torch.linalg.norm(self.ee_state[:,3:7] - goal_pos[:,3:7], axis = 1) 
        return  10*reach_cost + align_cost

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
        r = 0.08
        L = 2*0.157
        if self.robot == 'boxer':
            # Diff drive fk
            u_ik = u.clone()
            u_ik[:, 0] = (u[:, 0] / r) - ((L*u[:, 1])/(2*r))
            u_ik[:, 1] = (u[:, 0] / r) + ((L*u[:, 1])/(2*r))
            return u_ik
        if self.robot == 'albert':
            # Diff drive fk
            u_ik = u.clone()
            u_ik[:, 9] = (u[:, 9] / r) - ((L*u[:, 10])/(2*r))
            u_ik[:, 10] = (u[:, 9] / r) + ((L*u[:, 10])/(2*r))
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
        self.root_ort = actor_root_state[:, 3:7]
        
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
        elif self.robot == 'panda_no_hand':
            res = torch.clone(dof_states).view(-1, 14)
        elif self.robot == 'omni_panda':
            res = torch.clone(dof_states).view(-1, 24)
            self.omni_dofs = torch.clone(res)
        elif self.robot == 'shadow_hand':
            res = torch.clone(dof_states).view(-1, 48)
        elif self.robot == 'albert':
            res = torch.clone(dof_states).view(-1, 22)
            
            # For boxer
            res_ = actor_root_state[self.actors_per_env-1::self.actors_per_env]
            res_boxer = torch.cat([res_[:, 0:2], res_[:, 7:9]], axis=1)
            res[:,18:22] = res_boxer 
            # print(res_boxer.size())
            
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
        elif self.robot == 'panda_no_hand':
            state_pos = torch.cat((state[:, 0].unsqueeze(1), state[:, 2].unsqueeze(1), state[:, 4].unsqueeze(1),
                                   state[:, 6].unsqueeze(1), state[:, 8].unsqueeze(1), state[:, 10].unsqueeze(1),
                                   state[:, 12].unsqueeze(1)), 1)
        elif self.robot == 'omni_panda':
            state_pos = torch.cat((state[:, 0].unsqueeze(1), state[:, 2].unsqueeze(1), state[:, 4].unsqueeze(1),
                                   state[:, 6].unsqueeze(1), state[:, 8].unsqueeze(1), state[:, 10].unsqueeze(1),
                                   state[:, 12].unsqueeze(1), state[:, 14].unsqueeze(1), state[:, 16].unsqueeze(1),
                                   state[:, 18].unsqueeze(1), state[:, 20].unsqueeze(1), state[:, 22].unsqueeze(1)), 1)
        elif self.robot == 'albert':
            state_pos = torch.cat((state[:, 0].unsqueeze(1), state[:, 2].unsqueeze(1), state[:, 4].unsqueeze(1),
                                   state[:, 6].unsqueeze(1), state[:, 8].unsqueeze(1), state[:, 10].unsqueeze(1),
                                   state[:, 12].unsqueeze(1), state[:, 14].unsqueeze(1), state[:, 16].unsqueeze(1),
                                   state[:, 18].unsqueeze(1), state[:, 20].unsqueeze(1)), 1)   
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
        self._net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_cf = gymtorch.wrap_tensor(self._net_cf)
        self._net_cf = self.gym.refresh_net_contact_force_tensor(self.sim)
        self.net_cf_all = torch.sum(torch.abs(torch.cat((self.net_cf[:, 0].unsqueeze(1), self.net_cf[:, 1].unsqueeze(1), self.net_cf[:, 2].unsqueeze(1)), 1)),1)
        # Take only forces in x,y in modulus for each environment. Avoid all collisions
        self.net_cf = torch.sum(torch.abs(torch.cat((self.net_cf[:, 0].unsqueeze(1), self.net_cf[:, 1].unsqueeze(1)), 1)),1)
        # The last actors are allowed to collide with eachother (movabable obstacles and robot), check depending on the amount of actors
        if self.env_type == 'arena':   
            obst_up_to = 7
        elif self.env_type == 'lab':
            obst_up_to = 4 
        elif self.env_type == 'store':
            obst_up_to = 2
        elif self.env_type == 'shadow':
            obst_up_to = -1
        elif self.env_type == 'storm':
            obst_up_to = 4

        if obst_up_to > 0:
            if self.env_type == 'storm':
                coll_cost = 1000*torch.sum(self.net_cf.reshape([self.num_envs, int(self.net_cf.size(dim=0)/self.num_envs)])[:,0:obst_up_to], 1)
            else:
                coll_cost = 0.01*torch.sum(self.net_cf_all.reshape([self.num_envs, int(self.net_cf_all.size(dim=0)/self.num_envs)])[:,0:obst_up_to], 1)
        else:
            coll_cost = 0*torch.sum(self.net_cf.reshape([self.num_envs, int(self.net_cf.size(dim=0)/self.num_envs)])[:,0:obst_up_to], 1)

        # add collision cost fingers not ro
        # if self.robot == 'panda' or self.robot == 'omni_panda':
        #     gripper_force_cost = torch.sum(0.01*net_cf.reshape([self.num_envs, int(net_cf.size(dim=0)/self.num_envs)])[:,self.bodies_per_env-2:-1],1)
        #     coll_cost += gripper_force_cost
        
        w_c = 1000 # Weight for collisions
        # Binary check for collisions. So far checking all collision with unmovable obstacles. Movable obstacles touching unmovable ones are considered collisions       
        #coll_cost[coll_cost<=0.1] = 0
        
        if self.robot == 'boxer':
            #task_cost = self.get_navigation_cost(state[:, :2])
            task_cost = self.get_push_cost(state[:, :2])
        elif self.robot == 'point_robot':
            task_cost = self.get_push_cost(state_pos)
            #task_cost = self.get_push_not_goal_cost(state_pos)
            #task_cost = self.get_navigation_cost(state_pos)
            #task_cost = self.get_pull_cost(state_pos)
        elif self.robot == 'heijn':
            #task_cost = self.get_navigation_cost(state_pos)
            task_cost = self.get_push_cost(state_pos)
        elif self.robot == 'panda':
            #task_cost = self.get_panda_cost(state_pos)
            task_cost = self.get_panda_reach_cost(state_pos)
        elif self.robot == 'panda_no_hand':
            task_cost = self.get_panda_push_cost(state_pos)
        elif self.robot == 'omni_panda':
            task_cost = self.get_panda_cost(state_pos)
        elif self.robot == 'shadow_hand':
            task_cost = self.get_shadow_cost_2()
        elif self.robot == 'albert':
            task_cost = self.get_albert_cost(state_pos)

        # Acceleration cost
        acc_cost = 0.00001*torch.linalg.norm(torch.square((u[0:1]-past_u[0:1])/0.05), dim=1)
        
        if self.robot == 'panda' or 'omni_panda':
            acc_cost = 0.001*torch.linalg.norm(torch.square((u-past_u)/0.05), dim=1)
        
        past_u = torch.clone(u)
        
        
        return  task_cost + coll_cost #+ acc_cost # + w_u*control_cost