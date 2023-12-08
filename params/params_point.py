import torch

# Paramters in the reactive_tamp file
allow_viewer = False
task = "reactive"                   # "simple", "patrolling", "reactive"
num_envs = 200
nx = 4
noise_sigma = torch.tensor([[2, 0], [0, 2]], device="cuda:0", dtype=torch.float32) # 2 seems better for two corner
# noise_sigma = torch.tensor([[3, 0], [0, 3]], device="cuda:0", dtype=torch.float32) # 3 seems better for one corner
horizon = 15 # 12
lambda_ = 0.8
device = "cuda:0" 
u_max = torch.tensor([2.5, 2.5], device="cuda:0", dtype=torch.float32) # 2.5
u_min = torch.tensor([-2.5, -2.5], device="cuda:0", dtype=torch.float32) # 3 hybrid one corner becomes push
step_dependent_dynamics = True
terminal_state_cost = None
sample_null_action = True
use_priors = False
u_per_command = 15
filter_u = True

# Parameters in the sim file
sim_allow_viewer = True
sim_num_envs = 1 
dt = 0.05
block_index = 7
kp_suction = 400

# Parameters in common
robot = "point_robot"                      # choose from "point_robot", "boxer", "albert"
environment_type = "normal"                # choose from "normal", "battery"
spacing = 10.0
manual_control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
suction_active = False                     # the same with use_vacuum
print_flag = False