import torch

# Paramters in the reactive_tamp file
allow_viewer = False
task = "simple"                   # "simple", "patrolling", "reactive"
num_envs = 200
nx = 22
tensor_args = {'device':"cuda:0", 'dtype':torch.float32} 
noise_sigma = 2 * torch.eye(13, **tensor_args)
noise_sigma[7, 7] = 0.8  # finger 
noise_sigma[8, 8] = 0.8  # finger
noise_sigma[11, 11] = 5   # forward
noise_sigma[12, 12] = 6 # theta
horizon = 12
lambda_ = 0.01
u_max = 2 * torch.ones(13, **tensor_args)
u_max[7] = 0.5
u_max[8] = 0.5
u_max[11] = 5
u_max[12] = 6
u_min = -u_max
step_dependent_dynamics = True
terminal_state_cost = None
sample_null_action = True
use_priors = False
u_per_command = 12
filter_u = True

# Parameters in the sim file
sim_allow_viewer = True
sim_num_envs = 1 
dt = 0.01
block_index = "None"
kp_suction = 400

# Parameters in common
robot = "albert"                      # choose from "point_robot", "boxer", "albert", "panda"
environment_type = "albert_arena"                # choose from "normal", "battery", "cube"
spacing = 2.0
manual_control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
suction_active = False                     # the same with use_vacuum
print_flag = False