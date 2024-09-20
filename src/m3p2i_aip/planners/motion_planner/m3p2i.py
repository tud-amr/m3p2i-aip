import torch
from m3p2i_aip.utils import mppi_utils
import m3p2i_aip.planners.motion_planner.mppi as mppi

class M3P2I(mppi.MPPI):
    def __init__(self, cfg, dynamics=None, running_cost=None):
        super().__init__(cfg, dynamics, running_cost)
        self.suction_active = cfg.suction_active

    def update_gripper_command(self, task):
        if task in ["reach", "place"]:
            self.gripper_command = "open"
        elif task == "pick":
            self.gripper_command = "close"

    def get_pull_preference(self):
        if self.multi_modal:
            weight_push = torch.sum(self.weights[:self.half_K]).item()
            weight_pull = torch.sum(self.weights[self.half_K:]).item()
            return int(weight_pull > weight_push)
        else:
            return self.suction_active
        
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

        eta_1, exp_1 = self.update_infinite_beta(total_costs_1, self.beta_1, 10, 3)
        eta_2, exp_2 = self.update_infinite_beta(total_costs_2, self.beta_2, 10, 3)
        eta, exp_ = self.update_infinite_beta(total_costs, self.beta, 10, 3)

        self.weights_1 = 1 / eta_1 * exp_1 
        self.weights_2 = 1 / eta_2 * exp_2
        self.weights = 1 / eta * exp_ 
    
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