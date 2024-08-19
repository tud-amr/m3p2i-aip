from scipy import signal
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import torch, logging, functools, numpy as np, scipy.interpolate as si
from torch.distributions.multivariate_normal import MultivariateNormal
from m3p2i_aip.utils.skill_utils import _ensure_non_zero, is_tensor_like, bspline
from m3p2i_aip.utils.mppi_utils import generate_gaussian_halton_samples, scale_ctrl, cost_to_go
logger = logging.getLogger(__name__)

@dataclass
class MPPIConfig(object):
    """
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param nx: state dimension
        :param mppi_mode: 'halton-spline' or 'simple' corresponds to the type of mppi.
        :param sampling_method: 'halton' or 'random', sampling strategy while using mode 'halton-spline'. In 'simple', random sampling is forced to 'random' 
        :param noise_sigma: variance per action (nu x nu, assume v_t ~ N(u_t, noise_sigma))
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean        
        :param device: pytorch device
        :param lambda_: inverse temperature, positive scalar where smaller values will allow more exploration
        :param update_lambda: flag for updating inv temperature
        :param update_cov: flag for updating covariance
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param rollout_var_discount: Discount cost over control horizon
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param use_priors: Whether or not to use other prior controllers
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost   
    """

    num_samples: int = 200
    horizon: int = 12
    nx: int = 4
    mppi_mode: str = 'halton-spline'
    sampling_method: str = "halton"
    noise_sigma: Optional[List[List[float]]] = None
    noise_mu: Optional[List[float]] = None
    device: str = "cuda:0"
    lambda_: float = 1.0
    update_lambda: bool = False
    update_cov: bool = False
    u_min: Optional[List[float]] = None
    u_max: Optional[List[float]] = None
    u_init: float = 0.0
    U_init: Optional[List[List[float]]] = None
    u_scale: float = 1
    u_per_command: int = 1
    rollout_var_discount: float = 0.95
    sample_null_action: bool = False
    sample_previous_plan: bool = True
    sample_other_priors: bool = False
    noise_abs_cost: bool = False
    filter_u: bool = False
    use_priors: bool = False
    seed_val: int = 0
    eta_u_bound: int = 10
    eta_l_bound: int = 5

class MPPI():
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories thus it scales with the number of samples K. 

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning'  
    and 'STORM: An Integrated Framework for Fast Joint-Space Model-Predictive Control for Reactive Manipulation'

    Code based off https://github.com/UM-ARM-Lab/pytorch_mppi and https://github.com/NVlabs/storm

    This mppi can run in two modes: 'simple' and a 'halton-spline':
        - simple:           random sampling at each MPPI iteration from normal distribution with simple mean update. To use this set 
                            mppi_mode = 'simple_mean'
        - halton-spline:    samples only at the start a halton-spline which is then shifted according to the current moments of the control distribution. 
                            Moments are updated using gradient. To use this set
                            mppi_mode = 'halton-spline', sample_mode = 'halton'
                            Alternatively, one can also sample random trajectories at each iteration using gradient mean update by setting
                            mppi_mode = 'halton-spline', sample_mode = 'random'
    """

    def __init__(self, cfg: MPPIConfig, dynamics: Callable, running_cost: Callable):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        """
        self.env_type = cfg.env_type
        cfg = cfg.mppi

        # Utility vars
        self.K = cfg.num_samples
        self.half_K = int(self.K/2)
        self.T = cfg.horizon  
        self.filter_u = cfg.filter_u
        self.lambda_ = cfg.lambda_
        self.delta = None
        self.sample_null_action = cfg.sample_null_action
        self.u_per_command = cfg.u_per_command
        self.device = cfg.device
        self.tensor_args={'device':cfg.device, 'dtype':torch.float32}

        # Bound actions
        if cfg.u_max and not cfg.u_min:
            cfg.u_min = -cfg.u_max
        if cfg.u_min and not cfg.u_max:
            cfg.u_max = -cfg.u_min
        self.u_min = cfg.u_min
        self.u_max = cfg.u_max
        self.u_scale = cfg.u_scale

        # Noise and input initialization
        self.noise_abs_cost = cfg.noise_abs_cost
        if not cfg.noise_sigma:
            cfg.noise_sigma = np.identity(int(cfg.nx/2)).tolist()
        assert all([len(cfg.noise_sigma[0]) == len(row) for row in cfg.noise_sigma])
        if not cfg.noise_mu:
            cfg.noise_mu = [0.0] * len(cfg.noise_sigma)
        if not cfg.U_init:
            cfg.U_init = [[0.0] * len(cfg.noise_mu)] * cfg.horizon

        # Convert lists in cfg to tensors and put them on device
        self.noise_sigma = torch.tensor(cfg.noise_sigma, device=cfg.device)
        self.noise_mu = torch.tensor(cfg.noise_mu, device=cfg.device)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(
            self.noise_mu, covariance_matrix=self.noise_sigma
        )
        self.u_init = torch.tensor(cfg.u_init, device=cfg.device)
        # self.U = torch.tensor(cfg.U_init, device=cfg.device)  # !!!!
        self.U = self.noise_dist.sample((self.T,))
        self.u_max = torch.tensor(cfg.u_max, device=cfg.device)
        self.u_min = torch.tensor(cfg.u_min, device=cfg.device)

        # Dimensions of state nx and control nu
        self.nx = cfg.nx
        self.nu = 1 if len(self.noise_sigma.shape) == 0 else self.noise_sigma.shape[0]

        # Handle 1D edge case
        if self.nu == 1:
            self.noise_mu = self.noise_mu.view(-1)
            self.noise_sigma = self.noise_sigma.view(-1, 1)

        # Input initialization
        self.mean_action = torch.zeros((self.T, self.nu), **self.tensor_args)
        self.best_traj = self.mean_action.clone()
        self.best_traj_1 = self.mean_action.clone()
        self.best_traj_2 = self.mean_action.clone()
        self.mean_action_1 = torch.zeros((self.T, self.nu), **self.tensor_args)
        self.mean_action_2 = torch.zeros((self.T, self.nu), **self.tensor_args)

        # Costs and dynamics initialization
        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = None

        # Sampled results from last command
        self.state = None
        self.cost_total = None
        self.cost_total_non_zero = None
        self.weights = None
        self.states = None
        self.actions = None

        # Halton sampling 
        self.knot_scale = 4             # From mppi config storm
        self.seed_val = 0               # From mppi config storm
        self.n_knots = self.T//self.knot_scale
        self.ndims = self.n_knots * self.nu
        self.degree = 2                # From sample_lib storm
        self.Z_seq = torch.zeros(1, self.T, self.nu, **self.tensor_args)
        self.cov_action = torch.diagonal(self.noise_sigma, 0)
        self.scale_tril = torch.sqrt(self.cov_action)
        self.squash_fn = 'clamp'
        self.step_size_mean = 0.98      # From storm

        # Discount
        self.gamma = cfg.rollout_var_discount 
        self.gamma_seq = torch.cumprod(torch.tensor([1.0] + [self.gamma] * (self.T - 1)),dim=0).reshape(1, self.T)
        self.gamma_seq = self.gamma_seq.to(**self.tensor_args)
        self.beta = 1 # param storm
        self.beta_1 = 1
        self.beta_2 = 1
        self.beta = 1

        # Filtering
        self.sgf_window = 9
        self.sgf_order = 2
        if (self.sgf_window % 2) == 0:
            self.sgf_window -=1       # Some versions of the sav-go filter require odd window size

        # Lambda update, for now the update of lambda is not performed
        self.eta_max = 0.1      # 10%
        self.eta_min = 0.01     # 1%
        self.lambda_mult = 0.1  # Update rate

        # covariance update
        self.update_cov = cfg.update_cov   # !! weird if set to True
        self.step_size_cov = 0.7
        self.kappa = 0.005
    
    def set_mode(self, mppi_mode, sample_method, multi_modal):
        self.mppi_mode = mppi_mode
        self.sample_method = sample_method
        self.multi_modal = multi_modal and mppi_mode == 'halton-spline'

    def _dynamics(self, state, u, t=None):
        return self.F(state, u, t=None)

    def _running_cost(self, state):
        return self.running_cost(state)

    def command(self, state):
        """
            Given a state, returns the best action sequence
        """
        
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(**self.tensor_args)

        if self.mppi_mode == 'simple':
            self.U = torch.roll(self.U, -1, dims=0)

            cost_total = self._compute_total_cost_batch_simple() # [K]

            beta = torch.min(cost_total)
            self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)

            eta = torch.sum(self.cost_total_non_zero)
            self.weights = (1. / eta) * self.cost_total_non_zero # [K]
            
            self.U += torch.sum(self.weights.view(-1, 1, 1) * self.noise, dim=0) # [K, 1, 1] * [K, T, nu] --> [T, nu] sum over K

            action = self.U[:self.u_per_command]

        elif self.mppi_mode == 'halton-spline':
            # shift command 1 time step [T, nu]
            self.mean_action = self._shift_action(self.mean_action)
            if self.multi_modal:
                self.mean_action_1 = self._shift_action(self.mean_action_1)
                self.mean_action_2 = self._shift_action(self.mean_action_2)
                self.best_traj_1 = self._shift_action(self.best_traj_1)
                self.best_traj_2 = self._shift_action(self.best_traj_2)

            cost_total = self._compute_total_cost_batch_halton()
            action = torch.clone(self.mean_action) # !!
        
        # Compute top n trajs
        self.top_values, self.top_idx = torch.topk(self.weights, 20)
        if self.ee_states != 'None':
            self.top_trajs = torch.index_select(self.ee_states, 0, self.top_idx)
        else:
            self.top_trajs = torch.index_select(self.states, 0, self.top_idx)
            pos_idx = torch.tensor([0, 2], device=self.device, dtype=torch.int32)
            self.top_trajs = torch.index_select(self.top_trajs, 2, pos_idx)

        # Smoothing with Savitzky-Golay filter
        if self.filter_u:
            u_ = action.cpu().numpy()
            u_filtered = signal.savgol_filter(u_, self.sgf_window, self.sgf_order, deriv=0, delta=1.0, axis=0, mode='interp', cval=0.0)
            if self.device == "cpu":
                action = torch.from_numpy(u_filtered).to('cpu')
            else:
                action = torch.from_numpy(u_filtered).to('cuda')
        return action
    
    def _shift_action(self, action_seq):
        """
            Given an action_seq [T, nu], make a time shifted sequence
        """
        saved_action = action_seq[-1]
        action_seq = torch.roll(action_seq, -1, dims=0)
        action_seq[-1] = saved_action
        return action_seq

    def _compute_rollout_costs(self, perturbed_actions):
        """
            Given a sequence of perturbed actions, forward simulates their effects and calculates costs for each rollout
        """
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, **self.tensor_args)
        cost_horizon = torch.zeros([K, T], **self.tensor_args)
        cost_samples = cost_total

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)

        states = []
        actions = []
        ee_states = []

        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t]

            # Last rollout is a braking manover
            if self.sample_null_action:
                u[self.K -1, :] = torch.zeros_like(u[self.K -1, :])
                self.perturbed_action[self.K - 1][t] = u[self.K -1, :]

            state, u = self._dynamics(state, u, t)
            c = self._running_cost(state) # every time stes you get nsamples cost, we need that as output for the discount factor
            # Update action if there were changes in M3P2I due for instance to suction constraints
            self.perturbed_action[:,t] = u
            cost_samples += c
            cost_horizon[:, t] = c 

            # Save total states/actions
            states.append(state)
            actions.append(u)
            ee_state = 'None' #(self.ee_l_state[:, :3] + self.ee_r_state[:, :3])/2 if self.ee_l_state != 'None' else 'None'
            ee_states.append(ee_state) if ee_state != 'None' else []
            
        # Actions is K x T x nu
        # States is K x T x nx
        actions = torch.stack(actions, dim=-2)
        states = torch.stack(states, dim=-2)
        ee_states = torch.stack(ee_states, dim=-2) if ee_states != [] else 'None'

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples += c
        cost_total += cost_samples.mean(dim=0)
        
        if self.mppi_mode == 'halton-spline':
            if self.multi_modal:
                self.noise = self._update_multi_modal_distribution(cost_horizon, actions)
            else:
                self.noise = self._update_distribution(cost_horizon, actions)
        return cost_total, states, actions, ee_states

    #################### Random Sampling ####################
    def _compute_total_cost_batch_simple(self):
        """
            Samples random noise and computes perturbed action sequence at each iteration. Returns total cost
        """
        # Resample noise each time we take an action
        self.noise = self.noise_dist.sample((self.K, self.T))
        # Broadcast own control to noise over samples; now it's K x T x nu
        self.perturbed_action = self.U + self.noise
        
        # Naively bound control
        self.perturbed_action = self._bound_action(self.perturbed_action)
        if self.env_type == 'panda_env':
            self.perturbed_action[:, :, 8] = self.perturbed_action[:, :, 7]

        self.cost_total, self.states, self.actions, self.ee_states = self._compute_rollout_costs(self.perturbed_action)
        self.actions /= self.u_scale

        # Bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U

        action_cost = self.get_action_cost()

        # Action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2)) # [K, T, nu] * [K, T, nu] --> [K] sum over T and nu
        self.cost_total += perturbation_cost
        return self.cost_total

    def get_action_cost(self):
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv # Like original paper
        return action_cost

    def _bound_action(self, action):
        if self.u_max is not None:
            action = torch.max(torch.min(action, self.u_max), self.u_min)
        return action

    #################### Halton Sampling ####################
    def _compute_total_cost_batch_halton(self):
        """
            Samples Halton splines once and then shifts mean according to control distribution. If random sampling is selected 
            then samples random noise at each step. Mean of control distribution is updated using gradient
        """
        if self.sample_method == 'random':
            self.delta = self.get_samples(self.K, base_seed=0)
        elif self.delta == None and self.sample_method == 'halton':
            self.delta = self.get_samples(self.K, base_seed=0)

        # Add zero-noise seq so mean is always a part of samples
        self.delta[-1,:,:] = self.Z_seq
        # Keeps the size but scales values
        scaled_delta = torch.matmul(self.delta, torch.diag(self.scale_tril)).view(self.delta.shape[0], self.T, self.nu)

        # First time mean is zero then it is updated in the distribution
        if self.multi_modal:
            act_seq_1 = self.mean_action_1 + scaled_delta[:self.half_K, :, :]
            act_seq_2 = self.mean_action_2 + scaled_delta[self.half_K:, :, :]
            act_seq = torch.cat((act_seq_1, act_seq_2), 0)
        else:
            act_seq = self.mean_action + scaled_delta

        # Scales action within bounds. act_seq is the same as perturbed actions
        act_seq = scale_ctrl(act_seq, self.u_min, self.u_max, squash_fn=self.squash_fn)
        # print(act_seq.size())

        if self.multi_modal:
            act_seq[0, :, :] = self.best_traj_1
            act_seq[self.half_K, :, :] = self.best_traj_2
        
        self.perturbed_action = torch.clone(act_seq)
        if self.env_type == 'panda_env':
            self.perturbed_action[:, :, 8] = self.perturbed_action[:, :, 7]

        self.cost_total, self.states, self.actions, self.ee_states = self._compute_rollout_costs(self.perturbed_action)

        self.actions /= self.u_scale

        action_cost = self.get_action_cost()

        # Action perturbation cost
        perturbation_cost = torch.sum(self.mean_action * action_cost, dim=(1, 2))
        # if not self.multi_modal:
        #     self.cost_total += perturbation_cost
        return self.cost_total

    def _exp_util(self, costs):
        """
           Calculate weights using exponential utility given cost
           Iuput: costs [K, T], costs within horizon
        """
        traj_costs = cost_to_go(costs, self.gamma_seq) # [K, T]
        traj_costs = traj_costs[:,0] # [K] Costs for the next timestep
        total_costs = traj_costs - torch.min(traj_costs) #!! different from storm
        
        # Normalization of the weights
        exp_ = torch.exp((-1.0/self.beta) * total_costs)
        eta = torch.sum(exp_)       # tells how many significant samples we have, more or less
        self.weights = 1 / eta * exp_  # [K]
        # print('eta', eta)

        # Update beta to make eta converge within the bounds 
        if self.env_type == 'panda_env': # grady's thesis
            eta_u_bound = 20
            eta_l_bound = 10
            beta_lm = 0.9
            beta_um = 1.2
            if eta > eta_u_bound:
                self.beta = self.beta*beta_lm
            elif eta < eta_l_bound:
                self.beta = self.beta*beta_um
        
        self.total_costs = total_costs

    def get_samples(self, sample_shape, **kwargs): 
        """
            Gets as input the desired number of samples and returns the actual samples. 
            Depending on the method, the samples can be Halton or Random. Halton samples a 
            number of knots, later interpolated with a spline
        """
        if(self.sample_method=='halton'):   # !!
            self.knot_points = generate_gaussian_halton_samples(
                sample_shape,               # Number of samples
                self.ndims,                 # n_knots * nu (knots per number of actions)
                use_ghalton=True,
                seed_val=self.seed_val,     # seed val is 0 
                device=self.device,
                float_dtype=torch.float32)
            
            # Sample splines from knot points:
            # iteratre over action dimension:
            knot_samples = self.knot_points.view(sample_shape, self.nu, self.n_knots) # n knots is T/knot_scale (30/4 = 7)
            self.samples = torch.zeros((sample_shape, self.T, self.nu), **self.tensor_args)
            for i in range(sample_shape):
                for j in range(self.nu):
                    self.samples[i,:,j] = bspline(knot_samples[i,j,:], n=self.T, degree=self.degree)

        elif(self.sample_method == 'random'):
            self.samples = self.noise_dist.sample((self.K, self.T))
        
        return self.samples
 
    def _update_distribution(self, costs, actions):
        """
            Update moments using sample trajectories.
            So far only mean is updated, eventually one could also update the covariance
        """

        self._exp_util(costs)

        # Update best action
        best_idx = torch.argmax(self.weights)
        self.best_idx = best_idx
        self.best_traj = torch.index_select(actions, 0, best_idx).squeeze(0)
       
        weighted_seq = self.weights.view(-1, 1, 1) * actions # [K, T, nu]
        new_mean = torch.sum(weighted_seq, dim=0)

        # Gradient update for the mean
        self.mean_action = (1.0 - self.step_size_mean) * self.mean_action +\
            self.step_size_mean * new_mean 
        # print(self.mean_action.size()) # [T, nu]
       
        delta = actions - self.mean_action.unsqueeze(0)

        #Update Covariance
        if self.update_cov:
            #Diagonal covariance of size AxA
            weighted_delta = self.weights * (delta ** 2).T
            # cov_update = torch.diag(torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0))
            cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0)
    
            self.cov_action = (1.0 - self.step_size_cov) * self.cov_action + self.step_size_cov * cov_update
            self.cov_action += self.kappa #* self.init_cov_action
            # self.cov_action[self.cov_action < 0.0005] = 0.0005
            self.scale_tril = torch.sqrt(self.cov_action)

        return delta