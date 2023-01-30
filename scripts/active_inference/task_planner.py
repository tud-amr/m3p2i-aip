import torch
import numpy as np
from utils import env_conf
from active_inference import ai_agent, adaptive_action_selection
from active_inference import isaac_int_req_templates, isaac_state_action_templates

class PLANNER_SIMPLE:
    def __init__(self) -> None:
        self.task = "None"
        self.curr_goal = "None"

    def update_plan(self, robot_pos):
        self.task = "navigation"
        self.curr_goal = torch.tensor([3, 3], device="cuda:0")

class PLANNER_PATROLLING(PLANNER_SIMPLE):
    def __init__(self, goals) -> None:
        self.goals = goals
        self.goal_id = 0
        self.curr_goal = self.goals[self.goal_id]
        self.task = "navigation"
    
    def update_plan(self, robot_pos):
        if torch.norm(robot_pos - self.curr_goal) < 0.1:
            self.goal_id += 1
            if self.goal_id >= self.goals.size(0):
                self.goal_id = 0
            self.curr_goal = self.goals[self.goal_id]

class PLANNER_AIF(PLANNER_SIMPLE):
    def __init__(self) -> None:
        PLANNER_SIMPLE.__init__(self)
        # Define the required mdp structures 
        mdp_battery = isaac_int_req_templates.MDPBattery() 
        # Define ai agent with related mdp structure to reason about
        self.ai_agent_internal = ai_agent.AiAgent(mdp_battery)
        # Set the preference for the battery 
        self.ai_agent_internal.set_preferences(np.array([[1.], [0], [0]])) # Fixed preference for battery ok, following ['ok', 'low', 'critcal'] 
        self.battery_factor = 0.5
        self.battery_level = 100

    # Battery simulation
    def battery_sim(self, robot_pos):
        if torch.norm(robot_pos - env_conf.docking_station_loc) < 0.5:
            self.battery_level += self.battery_factor
        else:
            self.battery_level -= self.battery_factor
        self.battery_level = max(0, self.battery_level)
        self.battery_level = min(100, self.battery_level)

    # Battery observation
    def get_battery_obs(self):
        if self.battery_level > 55: 
            obs_battery = 0  # Battery is ok
        elif self.battery_level > 35:
            obs_battery = 1  # Battery is low
        else:
            obs_battery = 2  # Battery is critical
        return obs_battery

    # Upadte the task planner
    def update_plan(self, robot_pos):
        self.battery_sim(robot_pos)
        obs_battery = self.get_battery_obs()
        outcome_internal, curr_action_internal = adaptive_action_selection.adapt_act_sel(self.ai_agent_internal, obs_battery)
        print('Measured battery level', self.battery_level)
        # print('The outcome from the internal requirements is', outcome_internal)
        print('The selected action from the internal requirements is', curr_action_internal)
        if curr_action_internal == 'go_recharge':
            self.task = 'go_recharge'
            self.curr_goal = env_conf.docking_station_loc