import torch
import logging
from isaacgym import gymtorch
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.mppi_utils import generate_gaussian_halton_samples, scale_ctrl, cost_to_go
import functools
import numpy as np
from scipy import signal
import scipy.interpolate as si

logger = logging.getLogger(__name__)

def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))

def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray

def bspline(c_arr, t_arr=None, n=100, degree=3):
    sample_device = c_arr.device
    sample_dtype = c_arr.dtype
    cv = c_arr.cpu().numpy()

    if(t_arr is None):
        t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
    else:
        t_arr = t_arr.cpu().numpy()
    spl = si.splrep(t_arr, cv, k=degree, s=0.5)
    xx = np.linspace(0, cv.shape[0], n)
    samples = si.splev(xx, spl, ext=3)
    samples = torch.as_tensor(samples, device=sample_device, dtype=sample_dtype)
    return samples

def handle_batch_input(func):
    """For func that expect 2D input, handle input that have more than 2 dimensions by flattening them temporarily"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # assume inputs that are tensor-like have compatible shapes and is represented by the first argument
        batch_dims = []
        for arg in args:
            if is_tensor_like(arg) and len(arg.shape) > 2:
                batch_dims = arg.shape[:-1]  # last dimension is type dependent; all previous ones are batches
                break
        # no batches; just return normally
        if not batch_dims:
            return func(*args, **kwargs)

        # reduce all batch dimensions down to the first one
        args = [v.view(-1, v.shape[-1]) if (is_tensor_like(v) and len(v.shape) > 2) else v for v in args]
        ret = func(*args, **kwargs)
        # restore original batch dimensions; keep variable dimension (nx)
        if type(ret) is tuple:
            ret = [v if (not is_tensor_like(v) or len(v.shape) == 0) else (
                v.view(*batch_dims, v.shape[-1]) if len(v.shape) == 2 else v.view(*batch_dims)) for v in ret]
        else:
            if is_tensor_like(ret):
                if len(ret.shape) == 2:
                    ret = ret.view(*batch_dims, ret.shape[-1])
                else:
                    ret = ret.view(*batch_dims)
        return ret

    return wrapper

class MPPI():
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories thus it scales with the number of samples K. 

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning'  
    and 'STORM: An Integrated Framework for Fast Joint-Space Model-Predictive Control for Reactive Manipulation'

    Code based off and https://github.com/NVlabs/storm

    This mppi can run in two modes: 'simple' and a 'halton-spline':
        - simple:           random sampling at each MPPI iteration from normal distribution with simple mean update. To use this set 
                            mppi_mode = 'simple_mean'
        - halton-spline:    samples only at the start a halton-spline which is then shifted according to the current moments of the control distribution. 
                            Moments are updated using gradient. To use this set
                            mppi_mode = 'halton-spline', sample_mode = 'halton'
                            Alternatively, one can also sample random trajectories at each iteration using gradient mean update by setting
                            mppi_mode = 'halton-spline', sample_mode = 'random'
    """

    def __init__(self, params, dynamics, running_cost, nx, noise_sigma, num_samples=100, horizon=15, device="cpu",
                 terminal_state_cost=None,
                 lambda_=1.,
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
                 use_vacuum=False,
                 robot='point_robot',
                 noise_abs_cost=False,
                 actors_per_env=None,
                 env_type='arena', 
                 bodies_per_env=None,
                 filter_u=True):
        """
        TODO: fix parameters description once fixed
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param lambda_: temperature, positive scalar where larger values will allow more exploration
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param step_dependent_dynamics: whether the passed in dynamics needs horizon step passed in (as 3rd arg)
        :param rollout_samples: M, number of state trajectories to rollout for each control trajectory
            (should be 1 for deterministic dynamics and more for models that output a distribution)
        :param rollout_var_cost: Cost attached to the variance of costs across trajectory rollouts
        :param rollout_var_discount: Discount of variance cost over control horizon
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param use_priors: Wheher or not to use other prior controllers
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost
        """

        # Parameters for mppi and sampling mode
        self.mppi_mode = 'simple'    # halton-spline, simple
        self.sample_method = 'random'       # halton, random

        # Utility vars
        self.num_envs = num_samples
        self.K = num_samples 
        self.half_K = int(num_samples/2)
        self.T = horizon  
        self.filter_u = filter_u
        self.lambda_ = lambda_
        self.tensor_args={'device':device, 'dtype':noise_sigma.dtype} 
        self.delta = None
        self.sample_null_action = sample_null_action
        self.u_per_command = u_per_command
        # Dimensions of state nx and control nu
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]

        # Noise initialization
        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.tensor_args['dtype'])

        # Handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        self.noise_mu = noise_mu.to(self.tensor_args['device'])
        self.noise_sigma = noise_sigma.to(self.tensor_args['device'])
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_abs_cost = noise_abs_cost

        # Random noise dist
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

        # Inpu initialization
        if u_init is None:
            u_init = torch.zeros_like(noise_mu)
            self.mean_action = torch.zeros((self.T, self.nu), device=self.tensor_args['device'], dtype=self.tensor_args['dtype'])
            self.best_traj = self.mean_action.clone()
            self.best_traj_1 = self.mean_action.clone()
            self.best_traj_2 = self.mean_action.clone()
            self.mean_action_1 = torch.zeros((self.T, self.nu), device=self.tensor_args['device'], dtype=self.tensor_args['dtype'])
            self.mean_action_2 = torch.zeros((self.T, self.nu), device=self.tensor_args['device'], dtype=self.tensor_args['dtype'])

        # Bound actions
        self.u_min = u_min
        self.u_max = u_max
        self.u_scale = u_scale

        if self.u_max is not None and self.u_min is None: # Make sure if any of them is specified, both are specified
            if not torch.is_tensor(self.u_max):
                self.u_max = torch.tensor(self.u_max)
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            if not torch.is_tensor(self.u_min):
                self.u_min = torch.tensor(self.u_min)
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.tensor_args['device'])
            self.u_max = self.u_max.to(device=self.tensor_args['device'])
        
        # Control sequence (T x nu)
        self.U = U_init
        self.u_init = u_init.to(self.tensor_args['device'])

        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))

        # Costs and dynamics initialization
        self.step_dependency = step_dependent_dynamics
        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost

        # Sampled results from last command
        self.state = None
        self.cost_total = None
        self.cost_total_non_zero = None
        self.weights = None
        self.states = None
        self.actions = None

        # Defaults
        # self.knot_scale = 4
        # self.degree = 2

        # Halton sampling 
        self.knot_scale = 4             # From mppi config storm
        self.seed_val = 0               # From mppi config storm
        self.n_knots = self.T//self.knot_scale
        self.ndims = self.n_knots * self.nu
        self.degree = 2                # From sample_lib storm
        self.Z_seq = torch.zeros(1, self.T, self.nu, **self.tensor_args)
        self.cov_action = torch.diagonal(noise_sigma, 0)
        self.scale_tril = torch.sqrt(self.cov_action)
        self.squash_fn = 'clamp'
        self.step_size_mean = 0.98      # From storm

        # Discount
        self.gamma = rollout_var_discount 
        self.gamma_seq = torch.cumprod(torch.tensor([1.0] + [self.gamma] * (self.T - 1)),dim=0).reshape(1, self.T)
        self.gamma_seq = self.gamma_seq.to(**self.tensor_args)
        self.beta = 1 # param storm
        self.beta_1 = 1
        self.beta_2 = 1

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
        self.update_cov = False   # !! weird if set to True
        self.step_size_cov = 0.7
        self.kappa = 0.005
    
    def set_mode(self, mppi_mode, sample_method, multi_modal):
        self.mppi_mode = mppi_mode
        self.sample_method = sample_method
        self.multi_modal = multi_modal and mppi_mode == 'halton-spline'

    @handle_batch_input
    def _dynamics(self, state, u, t):
        return self.F(state, u, t) if self.step_dependency else self.F(state, u)

    @handle_batch_input
    def _running_cost(self, state, u, t):
        return self.running_cost(state, u, t)

    def command(self, state):
        """
            Given a state, returns the best action sequence
        """
        
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.tensor_args['dtype'], device=self.tensor_args['device'])

        if self.mppi_mode == 'simple':
            self.U = torch.roll(self.U, -1, dims=0)

            cost_total = self._compute_total_cost_batch_simple() # [K]

            beta = torch.min(cost_total)
            self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)

            eta = torch.sum(self.cost_total_non_zero)
            self.weights = (1. / eta) * self.cost_total_non_zero # [K]
            
            self.U += torch.sum(self.weights.view(-1, 1, 1) * self.noise, dim=0) # [K, 1, 1] * [K, T, nu] --> [T, nu] sum over K

            action = self.U[:self.u_per_command]

            # Lambda update
            # eta_max = 10
            # eta_min = 1
            # lambda_mult = 0.1
            # if eta > eta_max*self.K:
            #     self.lambda_ = (1+lambda_mult)*self.lambda_
            # elif eta < eta_min*self.K:
            #     self.lambda_ = (1-lambda_mult)*self.lambda_

        elif self.mppi_mode == 'halton-spline':
            # shift command 1 time step [T, nu]
            self.mean_action = self._shift_action(self.mean_action)
            self.mean_action_1 = self._shift_action(self.mean_action_1)
            self.mean_action_1 = self._shift_action(self.mean_action_1)
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
            pos_idx = torch.tensor([0, 2], device="cuda:0", dtype=torch.int32)
            self.top_trajs = torch.index_select(self.top_trajs, 2, pos_idx)

        # Smoothing with Savitzky-Golay filter
        if self.filter_u:
            u_ = action.cpu().numpy()
            u_filtered = signal.savgol_filter(u_, self.sgf_window, self.sgf_order, deriv=0, delta=1.0, axis=0, mode='interp', cval=0.0)
            if self.tensor_args['device'] == "cpu":
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

        cost_total = torch.zeros(K, device=self.tensor_args['device'], dtype=self.tensor_args['dtype'])
        cost_horizon = torch.zeros([K, T], device=self.tensor_args['device'], dtype=self.tensor_args['dtype'])
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
            c = self._running_cost(state, u, t) # every time stes you get nsamples cost, we need that as output for the discount factor
            # Update action if there were changes in fusion mppi due for instance to suction constraints
            self.perturbed_action[:,t] = u
            cost_samples += c
            cost_horizon[:, t] = c 

            # Save total states/actions
            states.append(state)
            actions.append(u)
            ee_state = (self.ee_l_state[:, :3] + self.ee_r_state[:, :3])/2 if self.ee_l_state != 'None' else 'None'
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
                self.noise = self._update_multi_modal_distribution_new(cost_horizon, actions)
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
        if self.robot == 'panda':
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
        # print(self.delta[0, :, :])

        # Add zero-noise seq so mean is always a part of samples
        self.delta[-1,:,:] = self.Z_seq
        # Keeps the size but scales values
        scaled_delta = torch.matmul(self.delta, torch.diag(self.scale_tril)).view(self.delta.shape[0], self.T, self.nu)

        # First time mean is zero then it is updated in the distribution
        if self.multi_modal:
            act_seq_1 = self.mean_action_1 + scaled_delta[:self.half_K, :, :] # also time shift!!
            act_seq_2 = self.mean_action_2 + scaled_delta[self.half_K:, :, :]
            act_seq = torch.cat((act_seq_1, act_seq_2), 0)
        else:
            act_seq = self.mean_action + scaled_delta

        # Scales action within bounds. act_seq is the same as perturbed actions
        act_seq = scale_ctrl(act_seq, self.u_min, self.u_max, squash_fn=self.squash_fn)
        # print(act_seq.size())
        # act_seq[self.nu, :, :] = self.best_traj # !!?

        # New weight, pick yes
        # New weight move no
        # Old weight move two corner maybe yes, not sure, maybe tune the range of costheta
        # Old weight move one corner yes, good 
        # Old weight pick yes, good
        if self.multi_modal:
            act_seq[0, :, :] = self.best_traj_1
            act_seq[self.half_K, :, :] = self.best_traj_2
        
        self.perturbed_action = torch.clone(act_seq)
        if self.robot == 'panda':
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
        print('eta', eta)

        # Update beta to make eta converge within the bounds 
        if self.env_type == 'cube': # grady's thesis
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
                device=self.tensor_args['device'],
                float_dtype=self.tensor_args['dtype'])
            
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
        # !!
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

    def _multi_modal_exp_util(self, costs):
        """
           Calculate weights using exponential utility given cost
           Iuput: costs [K, T], costs within horizon
        """
        traj_costs = cost_to_go(costs, self.gamma_seq) # [K, T]
        traj_costs = traj_costs[:,0] # [K] Costs for the next timestep

        total_costs_1 = traj_costs[:self.half_K] - torch.min(traj_costs[:self.half_K])
        total_costs_2 = traj_costs[self.half_K:] - torch.min(traj_costs[self.half_K:])
        # print('1', total_costs_1)
        # print('2', total_costs_2)
        # Normalization of the weights
        exp_1 = torch.exp((-1.0/self.beta_1) * total_costs_1)
        exp_2 = torch.exp((-1.0/self.beta_2) * total_costs_2)
        eta_1 = torch.sum(exp_1)
        eta_2 = torch.sum(exp_2)
        print('eta1', eta_1)
        print('eta2', eta_2)
        if self.env_type == 'normal':
            # two corner
            self.beta_1 = self.update_beta(self.beta_1, eta_1, 10, 1) 
            self.beta_2 = self.update_beta(self.beta_2, eta_2, 10, 1)
        self.weights_1 = 1 / eta_1 * exp_1 
        self.weights_2 = 1 / eta_2 * exp_2
        self.weights = torch.cat((self.weights_1, self.weights_2), 0)
        self.total_costs = torch.cat((total_costs_1, total_costs_2), 0)
    
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
        # print('1', self.best_idx_1)
        # print('2', self.best_idx_2)
       
        weighted_seq = self.weights.view(-1, 1, 1) * actions # [K, T, nu]
        self.mean_action_1 = torch.sum(weighted_seq[:self.half_K], dim=0)
        self.mean_action_2 = torch.sum(weighted_seq[self.half_K:], dim=0)

        # Gradient update for the mean
        self.mean_action = (1.0 - self.step_size_mean) * self.mean_action +\
            self.step_size_mean/2 * self.mean_action_1 + self.step_size_mean/2 * self.mean_action_2
        # print(self.mean_action.size()) # [T, nu]
       
        delta = actions - self.mean_action.unsqueeze(0)

        return delta
    
    def _multi_modal_exp_util_new(self, costs):
        """
           Calculate weights using exponential utility given cost
           Iuput: costs [K, T], costs within horizon
        """
        traj_costs = cost_to_go(costs, self.gamma_seq) # [K, T]
        traj_costs = traj_costs[:,0] # [K] Costs for the next timestep

        total_costs_1 = traj_costs[:self.half_K] - torch.min(traj_costs[:self.half_K])
        total_costs_2 = traj_costs[self.half_K:] - torch.min(traj_costs[self.half_K:])
        total_costs = traj_costs - torch.min(traj_costs)
        # print('1', total_costs_1)
        # print('2', total_costs_2)
        # Normalization of the weights
        exp_1 = torch.exp((-1.0/self.beta_1) * total_costs_1)
        exp_2 = torch.exp((-1.0/self.beta_2) * total_costs_2)
        exp_ = torch.exp((-1.0/self.beta) * total_costs)
        eta_1 = torch.sum(exp_1)
        eta_2 = torch.sum(exp_2)
        eta = torch.sum(exp_)
        print('eta1', eta_1)
        print('eta2', eta_2)
        print('eta', eta)
        # Update beta to make eta converge within the bounds 
        if self.env_type == 'cube': # grady's thesis
            self.beta_1 = self.update_beta(self.beta_1, eta_1, eta_u_bound=25, eta_l_bound=5)
            self.beta_2 = self.update_beta(self.beta_2, eta_2, eta_u_bound=25, eta_l_bound=5)
        elif self.env_type == 'normal':
            # two corner
            self.beta_1 = self.update_beta(self.beta_1, eta_1, 10, 3) 
            self.beta_2 = self.update_beta(self.beta_2, eta_2, 10, 3)
            # # one corner
            # self.beta_1 = self.update_beta(self.beta_1, eta_1, 10, 3) 
            # self.beta_2 = self.update_beta(self.beta_2, eta_2, 10, 3)

        self.weights_1 = 1 / eta_1 * exp_1 
        self.weights_2 = 1 / eta_2 * exp_2
        self.weights = 1 / eta * exp_ 
        # print('weights', self.weights)
    
    def update_beta(self, beta, eta, eta_u_bound, eta_l_bound):
        beta_lm = 0.9
        beta_um = 1.2
        if eta > eta_u_bound:
            beta = beta * beta_lm
        elif eta < eta_l_bound:
            beta = beta * beta_um
        return beta
    
    def _update_multi_modal_distribution_new(self, costs, actions):
        """
            Update moments using sample trajectories.
            So far only mean is updated, eventually one could also update the covariance
        """

        self._multi_modal_exp_util_new(costs)

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