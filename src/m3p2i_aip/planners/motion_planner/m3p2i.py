import torch
from isaacgym import gymtorch, gymapi
from m3p2i_aip.utils import skill_utils, mppi_utils
import m3p2i_aip.planners.motion_planner.mppi as mppi

class M3P2I(mppi.MPPI):
    def __init__(self, cfg, dynamics=None, running_cost=None):
        super().__init__(cfg, dynamics, running_cost)

        self.kp_suction = cfg.kp_suction
        self.suction_active = cfg.suction_active
        # self.env_type = env_type
        
        # Additional variables for the environment 
        self.block_goal = torch.tensor([0, 0], **self.tensor_args)
        self.block_not_goal = torch.tensor([-2, 1], **self.tensor_args)
        self.nav_goal = torch.tensor([3, 3], device=self.device)
        self.task = "navigation"  # "navigation", "push", "pull", "push_not_goal"
    
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
            weight_push = torch.sum(self.weights[:int(self.K/2)]).item()
            weight_pull = torch.sum(self.weights[int(self.K/2):]).item()
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

    # @mppi.handle_batch_input
    # def _running_cost(self, state, u, t):
    # #     # if self.robot == 'albert':
    # #     #     return self.get_albert_cost()
    # #     if self.task == 'navigation' or self.task == 'go_recharge':
    # #         task_cost = self.get_navigation_cost()
    # #     elif self.task == 'push':
    # #         task_cost = self.get_push_cost()
    # #     elif self.task == 'pull':
    # #         task_cost = self.get_pull_cost(False) # 10
    # #     elif self.task == 'push_not_goal':
    # #         task_cost = self.get_push_not_goal_cost()
    # #     elif self.task == 'hybrid':
    # #         return torch.cat((self.get_push_cost()[:self.half_K], self.get_pull_cost(True)[self.half_K:]), dim=0)
    # #         # print('push cost', task_cost[:10])
    # #         # print('pull cost', task_cost[self.num_envs-10:])
    # #     elif self.task == 'pick':
    # #         # return self.get_panda_pick_cost(self.multi_modal) # for albert
    # #         task_cost = self.get_panda_pick_cost(self.multi_modal) # for panda
    # #     elif self.task == 'place':
    # #         return self.get_panda_place_cost()
    # #     else:
    # #         task_cost = 0

    # #     total_cost = task_cost + self.get_motion_cost(t)
    #     goal = torch.tensor([1., 1.], device=self.cfg.device)
    #     return  torch.linalg.norm(self.sim.robot_pos - goal, axis=1)