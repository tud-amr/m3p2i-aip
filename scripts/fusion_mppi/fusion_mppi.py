import fusion_mppi.mppi as mppi
import torch
from isaacgym import gymtorch
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
                    noise_abs_cost)
        self.gym = None
        self.sim = None
        self.num_envs = num_samples
    
    def update_gym(self, gym, sim):
        self.gym = gym
        self.sim = sim

    @mppi.handle_batch_input
    def _dynamics(self, state, u, t):
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(u))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        dof_states, _, _, _ = sim_init.acquire_states(self.gym, self.sim, print_flag=False)
        res = torch.clone(dof_states).view(-1, 4)
        return res

    @mppi.handle_batch_input
    def _running_cost(self, state, u):
        # State: for each environment, the current state containing position and velocity
        # Action: same but for control input
        
        state_pos = torch.cat((state[:, 0].unsqueeze(1), state[:, 2].unsqueeze(1)), 1)
        task_cost = torch.linalg.norm(state_pos - torch.tensor([3, -3], device="cuda:0"), axis=1)
        
        control_cost = torch.sum(torch.square(u),1)
        w_u = 0.01
        # Contact forces
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        net_cf = gymtorch.wrap_tensor(_net_cf)
        _net_cf = self.gym.refresh_net_contact_force_tensor(self.sim)
        # Take only forces in x,y in modulus for each environment. Avoid all collisions
        net_cf = torch.sum(torch.abs(torch.cat((net_cf[:, 0].unsqueeze(1), net_cf[:, 1].unsqueeze(1)), 1)),1)
        coll_cost = torch.sum(net_cf.reshape([self.num_envs, int(net_cf.size(dim=0)/self.num_envs)]), 1)
        w_c = 10000 # Weight for collisions
        # Binary check for collisions. So far checking all collision of all actors. TODO: check collision of robot body only       
        coll_cost[coll_cost>0.1] = 1
        coll_cost[coll_cost<=0.1] = 0
        
        return  task_cost + w_u*control_cost + w_c*coll_cost