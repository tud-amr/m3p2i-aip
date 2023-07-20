import torch

# Paramters in the reactive_tamp file
allow_viewer = False
task = "simple"                   # "simple", "patrolling", "reactive"
num_envs = 500
nx = 18
noise_sigma = 0.01 * torch.eye(9, device="cuda:0", dtype=torch.float32) #0.05
noise_sigma[7, 7] = 0.8 # 0.8 original no collision cost 
noise_sigma[8, 8] = 0.8
horizon = 8
lambda_ = 0.01
device = "cuda:0" 
u_max = 0.2 * torch.ones(9, device="cuda:0", dtype=torch.float32)
u_max[7:] = 0.5
u_min = -0.2 * torch.ones(9, device="cuda:0", dtype=torch.float32)
u_min[7:] = -0.5
step_dependent_dynamics = True
terminal_state_cost = None
sample_null_action = False
use_priors = False
u_per_command = 8
filter_u = False

# Parameters in the sim file
sim_allow_viewer = True
sim_num_envs = 1 
dt = 0.01
block_index = "None"
kp_suction = 400

# Parameters in common
robot = "panda"                      # choose from "point_robot", "boxer", "albert", "panda"
environment_type = "cube"                # choose from "normal", "battery", "cube"
spacing = 2.0
manual_control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
suction_active = False                     # the same with use_vacuum
print_flag = False