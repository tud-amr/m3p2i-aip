from hashlib import new
import math
from isaacgym import gymapi
from isaacgym import gymtorch
import torch
import numpy as np
import sys
sys.path.append('../')
from active_inference import ai_agent, adaptive_action_selection
from active_inference import isaac_int_req_templates, isaac_state_action_templates 
from utils import env_conf, sim_init

class Params:
    def __init__(self) -> None:
        # Make the environment and simulation
        self.allow_viewer = True
        self.num_envs = 1
        self.spacing = 10.0
        self.robot = "point_robot"          # "point_robot", "boxer", "husky", "albert", and "heijn"
        self.environment_type = "battery"   # "normal", "battery", "lab"
        self.control_type = "vel_control"   # "vel_control", "pos_control", "force_control"
        self.print_flag = False
        self.block_index = "None"
        self.suction_active = False

# ------------------------ Initialization ------------------------
# Make the environment and simulation
params = Params()
gym, sim, viewer, envs, robot_handles = sim_init.make(params.allow_viewer, params.num_envs, 
                                                      params.spacing, params.robot, 
                                                      params.environment_type, params.control_type)
num_bodies = gym.get_actor_rigid_body_count(envs[-1], robot_handles[-1])

# Controller tuning
Kp = 1
def apply_control(u):
    # Get robot position at current time
    states_dict = sim_init.acquire_states(gym, sim, params)
    dof_states = states_dict["dof_states"]
    num_dofs = states_dict["num_dofs"]
    pos = dof_states[:,0]
    vel_target = torch.zeros(num_dofs, dtype=torch.float32, device="cuda:0")

    # u = 1: go normal, u = 2: go_recharge
    if u == 1: 
        pass
    if u == 'go_recharge': 
        vel_target = Kp * (env_conf.docking_station_loc.repeat(params.num_envs) - pos)
    gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(vel_target))

# Battery simulation
battery_factor = 0.5
def battery_sim(battery_level):
    states_dict = sim_init.acquire_states(gym, sim, params)
    dof_states = states_dict["dof_states"]
    pos = dof_states[:,0]
    if torch.norm(pos - env_conf.docking_station_loc.repeat(params.num_envs)) < 0.5:
         new_level = battery_level + battery_factor
    else:
        new_level = battery_level - battery_factor        # We can make this proportional to the velocity + a low dischrge factor
    if new_level < 0:
        new_level = 0
    if new_level > 100:
        new_level = 100
    return new_level

# Battery observation
def get_battery_obs(battery_level):
    if battery_level > 55: 
        obs_battery = 0  # Battery is ok for active inference
        color = env_conf.color_vec_battery_ok
    elif battery_level > 35:
        obs_battery = 1  # Battery is low
        color = env_conf.color_vec_battery_low
    else:
        obs_battery = 2  # Battery is critical
        color = env_conf.color_vec_battery_critical
    for i in range(params.num_envs):
        for n in range(num_bodies):
            gym.set_rigid_body_color(envs[i], robot_handles[i], n, gymapi.MESH_VISUAL, color)
    return obs_battery

# -------------------- Decision making (Active inference) --------------------
# Agent 1
# Define the required mdp structures 
mdp_battery = isaac_int_req_templates.MDPBattery() 
# Define ai agent with related mdp structure to reason about
ai_agent_internal = ai_agent.AiAgent(mdp_battery)
# Set the preference for the battery 
ai_agent_internal.set_preferences(np.array([[1.], [0], [0]])) # Fixed preference for battery ok, following ['ok', 'low', 'critcal'] 

# Agent 2
# Define the required mdp structures
mdp_isAt = isaac_state_action_templates.MDPIsAt() 
# Define ai agent with related mdp structure to reason about
ai_agent_task = ai_agent.AiAgent(mdp_isAt)
# Set the preference for the task 
ai_agent_task.set_preferences(np.array([[1.], [0]])) # Fixed preference for the task, following ['at_goal', 'not_at_goal']
# A typical sequence for decision making, ideally this should be repeated at a certain frequency

# ------------------------------ Main loop ------------------------------
t_decision = 0
battery_level = 100
while not gym.query_viewer_has_closed(viewer):
    # Step simulation
    sim_init.step(gym, sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    # Change robot color according to battery level
    battery_level = battery_sim(battery_level)

    # Decision making loop every second
    if t_decision == 0 or t_decision > 100:
        o_battery = get_battery_obs(battery_level)
        
        #print('battery level', battery_level)
        #print('Battey observation', o_battery)
        # o_isAt = 1
        # outcome_task, curr_action_task = adaptive_action_selection.adapt_act_sel(ai_agent_task, o_isAt)
        outcome_internal, curr_action_internal = adaptive_action_selection.adapt_act_sel(ai_agent_internal, o_battery)
        t_decision = 0

        # Printouts
        #print('The selected action from the task is', curr_action_task)
        # print('The selected action from the internal requirements is', curr_action_internal)
        #print('The battery state is:',  ai_agent_internal._mdp.state_names[np.argmax(ai_agent_internal.get_current_state())])
        # print('The action is:', ai_agent_internal._mdp.action_names[u])
        # print('Measured battery level', battery_level)
        # print('The isAt state is:',  ai_agent_task._mdp.state_names[np.argmax(ai_agent_task.get_current_state())])
        # print('The action is:', ai_agent_task._mdp.action_names[ut])
        # if curr_action_internal is not 'idle_success':
        #     print('Free energy', sum(ai_agent_task.G))
    # TODO: now we have the two actions one for the task and one for the internal needs, with related expected free energies. When do we do what? 

    # Compute and apply control action accoridng to action selection outcome (TODO) Substitute with MPPI
    apply_control(curr_action_internal)
    
    # Step rendering
    t_decision = t_decision + 1
    sim_init.step_rendering(gym, sim, viewer)

# Destroy the simulation  
sim_init.destroy_sim(gym, sim, viewer)