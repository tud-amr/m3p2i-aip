import fusion_mppi.mppi as mppi
import torch
from isaacgym import gymtorch, gymapi
from utils import sim_init

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
                    robot_type='point_robot',
                    noise_abs_cost=False):
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
                    robot_type,
                    noise_abs_cost)
        self.gym = None
        self.sim = None
        self.num_envs = num_samples
        self.robot = robot_type
        self.kp_suction = 400
        self.suction_active = True
        self.bodies_per_env = 18

        # Additional variables for the environment
        self.block_index = 7   # Pushing purple blox, index according to simulation
        self.block_goal = torch.tensor([3, -3], device="cuda:0")
        self.block_not_goal = torch.tensor([-2, 1], device="cuda:0")
        self.nav_goal = torch.tensor([3, 3], device="cuda:0")

    def update_gym(self, gym, sim, viewer=None):
        self.gym = gym
        self.sim = sim
        self.viewer = viewer

    def get_navigation_cost(self, r_pos):
        return torch.linalg.norm(r_pos - self.nav_goal, axis=1)

    def get_boxer_push_cost(self, r_pos):
        block_pos = torch.cat((torch.split(torch.clone(self.root_positions[:,0:2]), int(torch.clone(self.root_positions[:,0:2]).size(dim=0)/self.num_envs))),1)[self.block_index,:].reshape(self.num_envs,2)

        robot_to_block = torch.linalg.norm(r_pos - block_pos, axis = 1)
        block_to_goal = torch.linalg.norm(self.block_goal - block_pos, axis = 1)

        cost = robot_to_block + block_to_goal 

        robot_to_goal = torch.linalg.norm(r_pos - self.block_goal, axis = 1)
        align_cost = torch.zeros_like(robot_to_goal)
        align_cost[block_to_goal > robot_to_goal] = 1

        cost += align_cost
        return cost

    def get_push_cost(self, r_pos):
        block_pos = torch.cat((torch.split(torch.clone(self.root_positions[:,0:2]), int(torch.clone(self.root_positions[:,0:2]).size(dim=0)/self.num_envs))),1)[self.block_index,:].reshape(self.num_envs,2)

        robot_to_block = torch.linalg.norm(r_pos - block_pos, axis = 1)
        block_to_goal = torch.linalg.norm(self.block_goal - block_pos, axis = 1)

        cost = robot_to_block + block_to_goal 

        robot_to_goal = torch.linalg.norm(r_pos - self.block_goal, axis = 1)
        align_cost = torch.zeros_like(robot_to_goal)
        align_cost[block_to_goal > robot_to_goal] = 1

        cost += align_cost
        return cost
    
    def get_pull_cost(self, r_pos):
        # TODO modify to have only backwards velocities here or somewhere else when pull cost is active
        block_pos = torch.cat((torch.split(torch.clone(self.root_positions[:,0:2]), int(torch.clone(self.root_positions[:,0:2]).size(dim=0)/self.num_envs))),1)[self.block_index,:].reshape(self.num_envs,2)

        robot_to_block = torch.linalg.norm(r_pos - block_pos, axis = 1)
        block_to_goal = torch.linalg.norm(self.block_goal - block_pos, axis = 1)

        cost = robot_to_block + block_to_goal 

        robot_to_goal = torch.linalg.norm(r_pos - self.block_goal, axis = 1)
        align_cost = torch.zeros_like(robot_to_goal)
        align_cost[block_to_goal > robot_to_goal] = 1

        cost += align_cost
        return cost

    def get_push_not_goal_cost(self, r_pos):
        block_pos = torch.cat((torch.split(torch.clone(self.root_positions[:,0:2]), int(torch.clone(self.root_positions[:,0:2]).size(dim=0)/self.num_envs))),1)[self.block_index,:].reshape(self.num_envs,2)
        non_goal_cost = torch.clamp((1/torch.linalg.norm(self.block_not_goal - block_pos,axis = 1)), min=0, max=10)
        return torch.linalg.norm(r_pos - block_pos, axis = 1) + non_goal_cost

    def calculate_suction(self, v1, v2):
        dir_vector = v1 - v2
        magnitude = 1/torch.linalg.norm(dir_vector, dim=1)
        magnitude = torch.reshape(magnitude,[1,self.num_envs,1])
        direction = dir_vector/magnitude

        force = (magnitude**2)*direction
        forces = torch.zeros((self.num_envs, self.bodies_per_env, 3), dtype=torch.float32, device='cuda:0', requires_grad=False)

        if self.suction_active:   # Start suction only when close
            for i in range(self.num_envs):
                if magnitude[0,i] > 2:
                    forces[i,self.block_index, 0]= -self.kp_suction*force[0][i,0]
                    forces[i,self.block_index, 1] = -self.kp_suction*force[0][i,1]
                    # Opposite force on the robot body
                    forces[i,-1, 0] = self.kp_suction*force[0][i,0]
                    forces[i,-1, 1] = self.kp_suction*force[0][i,1]

        forces = torch.clamp(forces, min=-500, max=500)

        return forces

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

    @mppi.handle_batch_input
    def _dynamics(self, state, u, t):

        # Use inverse kinematics if the MPPI action space is different than dof velocity space
        u = self._ik(u)

        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(u))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        actor_root_state = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.root_positions = actor_root_state[:, 0:3]
        dof_states, _, _, _ = sim_init.acquire_states(self.gym, self.sim, print_flag=False)

        dof_pos = torch.clone(dof_states).view(-1, 4)[:,[0,2]]
        print('dof_states', dof_pos)

        if self.suction_active:
            root_pos = torch.reshape(self.root_positions[:, 0:2], (self.num_envs, self.bodies_per_env-2, 2))
            #dof_pos = dof_states[:,[0,2]]
            # simulation of a magnetic/suction effect to attach to the box
            suction_force = self.calculate_suction(root_pos[:, self.block_index, :], dof_pos)
        
            # Apply suction/magnetic force
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(torch.reshape(suction_force, (self.num_envs*self.bodies_per_env, 3))), None, gymapi.ENV_SPACE)

        if self.robot == 'boxer':
            res_ = actor_root_state[13::14]
            res = torch.cat([res_[:, 0:2], res_[:, 7:9]], axis=1)
        elif self.robot == 'point_robot':
            res = torch.clone(dof_states).view(-1, 4)   

        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

        return res

    @mppi.handle_batch_input
    def _running_cost(self, state, u):
        # State: for each environment, the current state containing position and velocity
        # Action: same but for control input
        
        state_pos = torch.cat((state[:, 0].unsqueeze(1), state[:, 2].unsqueeze(1)), 1)
        control_cost = torch.sum(torch.square(u),1)
        w_u = 0.01
        # Contact forces
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        net_cf = gymtorch.wrap_tensor(_net_cf)
        _net_cf = self.gym.refresh_net_contact_force_tensor(self.sim)
        # Take only forces in x,y in modulus for each environment. Avoid all collisions
        net_cf = torch.sum(torch.abs(torch.cat((net_cf[:, 0].unsqueeze(1), net_cf[:, 1].unsqueeze(1)), 1)),1)
        # The last actors are allowed to collide with eachother (movabable obstacles and robot)
        coll_cost = torch.sum(net_cf.reshape([self.num_envs, int(net_cf.size(dim=0)/self.num_envs)])[:,0:6], 1)
        w_c = 1000 # Weight for collisions
        # Binary check for collisions. So far checking all collision with unmovable obstacles. Movable obstacles touching unmovable ones are considered collisions       
        coll_cost[coll_cost>0.1] = 1
        coll_cost[coll_cost<=0.1] = 0
        if self.robot == 'boxer':
            task_cost = self.get_boxer_push_cost(state[:, :2])
        elif self.robot == 'point_robot':
            #task_cost = self.get_push_cost(state_pos)
            #task_cost = self.get_push_not_goal_cost(state_pos)
            #task_cost = self.get_navigation_cost(state_pos)
            task_cost = self.get_pull_cost(state_pos)
        return  task_cost + w_c*coll_cost # + w_u*control_cost 