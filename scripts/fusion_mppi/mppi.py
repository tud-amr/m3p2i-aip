import torch
import time
import logging
from isaacgym import gymtorch
from torch.distributions.multivariate_normal import MultivariateNormal
import functools
import numpy as np
from priors.fabrics_planner import fabrics_point

logger = logging.getLogger(__name__)


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))


def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray


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
                 bodies_per_env=None):
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
        self.d = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS

        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        if u_init is None:
            u_init = torch.zeros_like(noise_mu)

        # handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        # bounds
        self.u_min = u_min
        self.u_max = u_max
        self.u_scale = u_scale
        self.u_per_command = u_per_command
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
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        # T x nu control sequence
        self.U = U_init
        self.u_init = u_init.to(self.d)

        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))

        self.step_dependency = step_dependent_dynamics
        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.use_priors = use_priors
        self.noise_abs_cost = noise_abs_cost
        self.state = None

        # handling dynamics models that output a distribution (take multiple trajectory samples)
        self.M = rollout_samples
        self.rollout_var_cost = rollout_var_cost
        self.rollout_var_discount = rollout_var_discount

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

        # Priors parameters
        self.kp = 0.5

        # Initialize fabrics prior
        if self.use_priors:
            # Create fabrics planner with single obstacle
            goal = [3.0, 3.0]
            weight = 1.0
            self._fabrics_prior = fabrics_point(goal, weight)

    @handle_batch_input
    def _dynamics(self, state, u, t):
        return self.F(state, u, t) if self.step_dependency else self.F(state, u)

    @handle_batch_input
    def _running_cost(self, state, u):
        return self.running_cost(state, u)

    def command(self, state):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :returns action: (nu) best action
        """
        # shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init

        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)

        cost_total = self._compute_total_cost_batch()

        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)

        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1. / eta) * self.cost_total_non_zero
        for t in range(self.T):
            self.U[t] += torch.sum(self.omega.view(-1, 1) * self.noise[:, t], dim=0)
        action = self.U[:self.u_per_command]
        # reduce dimensionality if we only need the first command
        if self.u_per_command == 1:
            action = action[0]

        return action

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = self.noise_dist.sample((self.T,))

    def _compute_rollout_costs(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total.repeat(self.M, 1)
        cost_var = torch.zeros_like(cost_total)

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)

        # rollout action trajectory M times to estimate expected cost
        state = state.repeat(self.M, 1, 1)

        states = []
        actions = []

        actor_root_state = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        root_positions = actor_root_state[:, 0:3]

        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t].repeat(self.M, 1, 1)
            
            # Last rollout is a breaking manover
            if self.sample_null_action:
                u[:, self.K -1, :] = torch.zeros_like(u[:, self.K -1, :])
                # Update perturbed action sequence for later use in cost computation
                self.perturbed_action[self.K - 1][t] = u[:, self.K -1, :]

            if self.use_priors:
                u = self._priors_command(state, u, t, root_positions)

            state, u = self._dynamics(state, u, t)
            c = self._running_cost(state, u)
            # Update action if there were changes in fusion mppi due for instance to suction constraints
            self.perturbed_action[:,t] = u

            cost_samples += c
            if self.M > 1:
                cost_var += c.var(dim=0) * (self.rollout_var_discount ** t)

            # Save total states/actions
            states.append(state)
            actions.append(u)

        # Actions is K x T x nu
        # States is K x T x nx
        actions = torch.stack(actions, dim=-2)
        states = torch.stack(states, dim=-2)

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples += c
        cost_total += cost_samples.mean(dim=0)
        cost_total += cost_var * self.rollout_var_cost
        return cost_total, states, actions

    def _fabrics_prior_command(self, pos, vel, goal, obst):
        acc_action = self._fabrics_prior.compute_action(
            q=pos,
            qdot=vel,
            x_goal_0=goal,
            weight_goal_0=2.,
            x_obst_0=obst,
            weight_obst_0=0.1,
            radius_obst_0=np.array([0.4]),
            radius_body_1=np.array([0.2])
        )
        if any(np.isnan(acc_action)):
            acc_action = np.zeros_like(acc_action)
        vel_action = torch.tensor(np.array(vel) + acc_action*0.05, dtype=torch.float32, device="cuda:0")
        return vel_action

    def _priors_command(self, state, u, t, root_positions):
        nav_goal = self.nav_goal.cpu().numpy()
        u[:, self.K - 2, :] = -self.kp*torch.tensor([[state[0, -2][0] - nav_goal[0], state[0, -2][2] - nav_goal[1]]])    # Proportional controller to the goal
        # Clamp control input
        u[:, self.K - 2, :][0] = torch.clamp(u[:, self.K - 2, :][0], min=self.u_min, max=self.u_max)
        self.perturbed_action[self.K - 2][t] = u[:, self.K -2, :]

        # fabrics priors 
        # TODO: make fabrics prior computation parallel instead of sequential for speedups.
        # specify fabrics goals as list of tuples: (frame_id, position)
        fabrics_goals = [
            ("world", nav_goal)
        ] + [("robot", np.random.uniform(low=-2, high=2, size=(2,))) for _ in range(10)]

        fabrics_obsts = [
            ("world", [20, 20])
        ] + [("world", [20, 20])]*10

        assert len(fabrics_goals) == len(fabrics_obsts)

        for i, (goal, obst) in enumerate(zip(fabrics_goals, fabrics_obsts)):
            dof_state_np = state[0, i].cpu().numpy()
            pos = np.array([dof_state_np[0], dof_state_np[2]])
            vel = np.array([dof_state_np[1], dof_state_np[3]])

            if goal[0] == 'world':
                goal_pos = goal[1]
            elif goal[0] == 'robot':
                goal_pos = goal[1] + pos
            elif goal[0] == 'block':
                block_pos = torch.cat((torch.split(torch.clone(root_positions[:,0:2]), int(torch.clone(root_positions[:,0:2]).size(dim=0)/self.num_envs))),1)[self.block_index,:].reshape(self.num_envs,2)[i]
                goal_pos = goal[1] + block_pos.cpu().numpy()
            else:
                raise NotImplementedError()

            if obst[0] == 'world':
                obst_pos = obst[1]
            elif obst[0] == 'robot':
                obst_pos = obst[1] + pos
            elif obst[0] == 'block':
                block_pos = torch.cat((torch.split(torch.clone(root_positions[:,0:2]), int(torch.clone(root_positions[:,0:2]).size(dim=0)/self.num_envs))),1)[self.block_index,:].reshape(self.num_envs,2)[i]
                obst_pos = obst[1] + block_pos.cpu().numpy()
            else:
                raise NotImplementedError()

            u[:, i, :] = self._fabrics_prior_command(pos, vel, np.array(goal_pos), np.array(obst_pos))
            u[:, i, :] = torch.clamp(u[:, i, :], min=self.u_min, max=self.u_max)
            self.perturbed_action[i][t] = u[:, i, :]

        return u

    def _compute_total_cost_batch(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        self.noise = self.noise_dist.sample((self.K, self.T))
        # broadcast own control to noise over samples; now it's K x T x nu
        self.perturbed_action = self.U + self.noise
        
        # naively bound control
        self.perturbed_action = self._bound_action(self.perturbed_action)

        self.cost_total, self.states, self.actions = self._compute_rollout_costs(self.perturbed_action)
        self.actions /= self.u_scale

        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U

        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv # Like original paper

        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total += perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.u_max is not None:
            action = torch.max(torch.min(action, self.u_max), self.u_min)
        return action

    def get_rollouts(self, state, num_rollouts=1):
        """
            :param state: either (nx) vector or (num_rollouts x nx) for sampled initial states
            :param num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic
                                 dynamics
            :returns states: num_rollouts x T x nx vector of trajectories

        """
        state = state.view(-1, self.nx)
        if state.size(0) == 1:
            state = state.repeat(num_rollouts, 1)

        T = self.U.shape[0]
        states = torch.zeros((num_rollouts, T + 1, self.nx), dtype=self.U.dtype, device=self.U.device)
        states[:, 0] = state
        for t in range(T):
            states[:, t + 1] = self._dynamics(states[:, t].view(num_rollouts, -1),
                                              self.u_scale * self.U[t].view(num_rollouts, -1), t)
        return states[:, 1:]


def run_mppi(mppi, env, retrain_dynamics, retrain_after_iter=50, iter=1000, render=True):
    dataset = torch.zeros((retrain_after_iter, mppi.nx + mppi.nu), dtype=mppi.U.dtype, device=mppi.d)
    total_reward = 0
    for i in range(iter):
        state = env.state.copy()
        command_start = time.perf_counter()
        action = mppi.command(state)
        elapsed = time.perf_counter() - command_start
        s, r, _, _ = env.step(action.cpu().numpy())
        total_reward += r
        logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r, elapsed)
        if render:
            env.render()

        di = i % retrain_after_iter
        if di == 0 and i > 0:
            retrain_dynamics(dataset)
            # don't have to clear dataset since it'll be overridden, but useful for debugging
            dataset.zero_()
        dataset[di, :mppi.nx] = torch.tensor(state, dtype=mppi.U.dtype)
        dataset[di, mppi.nx:] = action
    return total_reward, dataset
