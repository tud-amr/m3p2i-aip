import torch
import logging
from isaacgym import gymtorch
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.control_utils import generate_gaussian_halton_samples, scale_ctrl, cost_to_go
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
    count = len(cv)

    if(t_arr is None):
        t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
    else:
        t_arr = t_arr.cpu().numpy()
    spl = si.splrep(t_arr, cv, k=degree, s=0.5)
    # #spl = BSpline(t, c, k, extrapolate=False)
    xx = np.linspace(0, cv.shape[0], n)
    samples = si.splev(xx, spl, ext=3)
    samples = torch.as_tensor(samples, device=sample_device, dtype=sample_dtype)
    return samples

# from arm_pytorch_utilities, standalone since that package is not on pypi yet
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
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(self, dynamics, running_cost, nx, noise_sigma, num_samples=100, horizon=15, device="cpu",
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

        self.num_envs = num_samples
        self.robot = robot
        self.bodies_per_env = bodies_per_env
        self.d = device
        self.env_type = env_type
        self.dtype = noise_sigma.dtype
        self.K = num_samples 
        self.T = horizon  
        self.filter_u = filter_u
        self.lambda_ = lambda_
        self.tensor_args={'device':self.d, 'dtype':torch.float32} 
        self.delta = None
        self.sample_null_action = sample_null_action

        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]

        # Temp. visualization
        self.ee_indexes = np.zeros(self.num_envs)
        self.ee_index = -1

        for i in range(self.num_envs):
            self.ee_indexes[i] = self.ee_index + i*self.bodies_per_env

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        if u_init is None:
            u_init = torch.zeros_like(noise_mu)
            self.mean_action = torch.zeros(self.nu, device=self.d, dtype=self.dtype)
            self.best_traj = self.mean_action.clone()

        # handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        # bounds
        self.u_min = u_min
        self.u_max = u_max
        self.u_scale = u_scale

        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            if not torch.is_tensor(self.u_max):
                self.u_max = torch.tensor(self.u_max)
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            if not torch.is_tensor(self.u_min):
                self.u_min = torch.tensor(self.u_min)
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        # Random noise dist, to be used if not Halton
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        
        # T x nu control sequence
        # self.U = U_init
        self.u_init = u_init.to(self.d)

        # if self.U is None:
        #     self.U = self.noise_dist.sample((self.T,))

        self.step_dependency = step_dependent_dynamics
        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost

        self.noise_abs_cost = noise_abs_cost
        self.state = None

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

        # Halton sampling, from storm
        self.knot_scale = 4     # From mppi config storm
        self.sample_method = 'halton-spline'
        self.seed_val = 0       # From mppi config storm
        self.n_knots = self.T//self.knot_scale
        self.ndims = self.n_knots * self.nu
        self.degree = 2       # hardcoded value in sample_lib storm
        self.Z_seq = torch.zeros(1, self.T, self.nu, **self.tensor_args)
        self.cov_action = torch.diagonal(noise_sigma, 0)
        self.scale_tril = torch.sqrt(self.cov_action)
        self.squash_fn = 'clamp'
        self.step_size_mean = 0.98

        # # Random
        # self.Z = torch.zeros(self.ndims, **self.tensor_args)
        # self.scale = torch.linalg.cholesky(self.noise_sigma.to(dtype=torch.float32)).to(**self.tensor_args)
        # self.mvn = MultivariateNormal(loc=self.Z, scale_tril=self.scale, )

        # Discount
        self.gamma = 0.95 # Param storm
        self.gamma_seq = torch.cumprod(torch.tensor([1.0] + [self.gamma] * (self.T - 1)),dim=0).reshape(1, self.T)
        self.gamma_seq = self.gamma_seq.to(**self.tensor_args)
        self.beta = 1 # param storm

        # filtering
        self.sgf_window = 9
        self.sgf_order = 2
        if (self.sgf_window % 2) == 0:
            self.sgf_window -=1       # Some versions of the sav-go filter require odd window size

        # Lambda update
        self.eta_max = 0.1      # 10%
        self.eta_min = 0.01     # 1%
        self.lambda_mult = 0.1  # Update rate

    @handle_batch_input
    def _dynamics(self, state, u, t):
        return self.F(state, u, t) if self.step_dependency else self.F(state, u)

    @handle_batch_input
    def _running_cost(self, state, u):
        return self.running_cost(state, u)

    def _exp_util(self, costs, actions):
        """
            storm Calculate weights using exponential utility
        """
        traj_costs = cost_to_go(costs, self.gamma_seq)
        # if not self.time_based_weights: traj_costs = traj_costs[:,0]
        traj_costs = traj_costs[:,0]
        #control_costs = self._control_costs(actions)

        total_costs = traj_costs #+ self.beta * control_costs
        
        
        # #calculate soft-max
        w = torch.softmax((-1.0/self.beta) * total_costs, dim=0)
        self.total_costs = total_costs
        return w

    def get_samples(self, sample_shape, **kwargs):      # sample shape is the number of rollouts
    #Looks like the halton samples are only generated once
    # sample shape is the number of particles to sample
        if(self.sample_method=='halton-spline'):
            self.knot_points = generate_gaussian_halton_samples(
                sample_shape,               # This is the number of samples
                self.ndims,                 # n_knots * nu (knots per number of actions)
                use_ghalton=True,
                seed_val=self.seed_val,     # seed val is 0 hardcoded
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

    def command(self, state):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :returns action: (nu) best action
        """
        # shift command 1 time step
        self.mean_action = torch.roll(self.mean_action, -1, dims=0)
        # Set first sequence to zero, otherwise it takes the last of the sequence
        self.mean_action[0].zero_()

        # TODO shift the best policy as well

        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)

        cost_total = self._compute_total_cost_batch()

        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)

        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1. / eta) * self.cost_total_non_zero
        
        # self.U += torch.sum(self.omega.view(-1, 1, 1) * self.noise, dim=0)
        
        action = torch.clone(self.mean_action)

        # # Lambda update
        # if eta > 5:
        #     self.lambda_ = (1-self.lambda_mult)*self.lambda_
        # elif eta < 2:
        #     self.lambda_ = (1+self.lambda_mult)*self.lambda_

        # Smoothing with Savitzky-Golay filter
        if self.filter_u:
            u_ = action.cpu().numpy()
            u_filtered = signal.savgol_filter(u_, self.sgf_window, self.sgf_order, deriv=0, delta=1.0, axis=0, mode='interp', cval=0.0)
            if self.d == "cpu":
                action = torch.from_numpy(u_filtered).to('cpu')
            else:
                action = torch.from_numpy(u_filtered).to('cuda')
        
        return action

    def _compute_rollout_costs(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_horizon = torch.zeros([K, T], device=self.d, dtype=self.dtype)
        cost_samples = cost_total
        cost_var = torch.zeros_like(cost_total)

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
            c = self._running_cost(state, u) # every time stes you get nsamples cost, we need that as output for the discount factor
            # Update action if there were changes in fusion mppi due for instance to suction constraints
            self.perturbed_action[:,t] = u
            cost_samples += c
            cost_horizon[:, t] = c 

            # Save total states/actions
            ee_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))[self.ee_indexes, 0:3]
            states.append(state)
            actions.append(u)
            ee_states.append(ee_state)
            
        # Actions is K x T x nu
        # States is K x T x nx
        actions = torch.stack(actions, dim=-2)
        states = torch.stack(states, dim=-2)
        ee_states = torch.stack(ee_states, dim=-2)

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples += c
        cost_total += cost_samples.mean(dim=0)
        
        #Update distributions here, line 105 mppi storm
        self.noise = self._update_distribution(cost_horizon, actions)

        return cost_total, states, actions, ee_states
 
    def _update_distribution(self, costs, actions):
        """
           Storm: Update moments in the direction using sampled
           trajectories

            So far only mean is updated, they do not update the covariance in storm franka example
        """

        # TODO compute also top n best actions to plot
        w = self._exp_util(costs, actions) # eq 6
        
        #Update best action
        best_idx = torch.argmax(w)
        self.best_idx = best_idx
        self.best_traj = torch.index_select(actions, 0, best_idx).squeeze(0)
       
        weighted_seq = w * actions.T

        sum_seq = torch.sum(weighted_seq.T, dim=0)   # Sum of weigths is 1, softmax

        new_mean = sum_seq
      
        self.mean_action = (1.0 - self.step_size_mean) * self.mean_action +\
            self.step_size_mean * new_mean # eq. 4
       
        delta = actions - self.mean_action.unsqueeze(0)
        return delta

    def _compute_total_cost_batch(self):
        # parallelize sampling across trajectories

        if self.sample_method == 'random':
            self.delta = self.get_samples(self.K, base_seed=0)
        # (Storm) Sample halton. Samples are done once at the beginning and then shifted considering
        # the mean and covariance of the gaussian policy 
        elif self.delta == None and self.sample_method == 'halton-spline':
            self.delta = self.get_samples(self.K, base_seed=0)
            #add zero-noise seq so mean is always a part of samples

        #add zero-noise seq so mean is always a part of samples
        self.delta[-1,:,:] = self.Z_seq
        #keeps the size but scales values
        scaled_delta = torch.matmul(self.delta, torch.diag(self.scale_tril)).view(self.delta.shape[0],
                                                                      self.T,
                                                                      self.nu)
        
        # For us the action sequence is self.U, for storm instead it is the mean

        # First time mean is zero then it is updated in the doistribution
        act_seq = self.mean_action + scaled_delta
        # mean should be updated after generating the rollouts and computing the optimal control

        #Scales action within bounds. act_seq is the same as perturbed actions
        act_seq = scale_ctrl(act_seq, self.u_min, self.u_max, squash_fn=self.squash_fn)
        act_seq[self.nu, :, :] = self.best_traj

        # TODO: Append best past trajectory (not necessary now) self.best_traj = self.mean_action.clone()
        # See mppi.py line 111 in storm, there they update the best trajectory and then append to current samples
        
        # resample noise each time we take an action
        # self.noise = self.noise_dist.sample((self.K, self.T))
        # broadcast own control to noise over samples; now it's K x T x nu
        self.perturbed_action = torch.clone(act_seq)
        # naively bound control
        # self.perturbed_action = self._bound_action(self.perturbed_action)

        self.cost_total, self.states, self.actions, self.ee_states = self._compute_rollout_costs(self.perturbed_action)

        # costs [ns, T], actions [ns, T, nu]


        self.actions /= self.u_scale

        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        #self.noise = self.perturbed_action - self.U

        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            #NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            #the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            #nomial trajectory.
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv # Like original paper

        # action perturbation cost
        perturbation_cost = torch.sum(self.mean_action * action_cost, dim=(1, 2))
        self.cost_total += perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.u_max is not None:
            action = torch.max(torch.min(action, self.u_max), self.u_min)
        return action
