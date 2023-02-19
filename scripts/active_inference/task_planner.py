import torch
import numpy as np
from utils import env_conf
from active_inference import ai_agent, adaptive_action_selection
from active_inference import isaac_int_req_templates, isaac_state_action_templates
import sys
import os
sys.path.append('../')
from utils import path_utils

class PLANNER_SIMPLE:
    def __init__(self) -> None:
        self.task = "None"
        self.curr_goal = "None"

    def update_plan(self, robot_pos, stay_still):
        self.task = "navigation"
        self.curr_goal = torch.tensor([3, 3], device="cuda:0")
    
    def reset_plan(self):
        self.task = "None"
        self.curr_goal = "None"     

class PLANNER_PATROLLING(PLANNER_SIMPLE):
    def __init__(self, goals) -> None:
        self.task = "navigation"
        self.goals = goals
        self.goal_id = 0
        self.curr_goal = self.goals[self.goal_id]
    
    def reset_plan(self):
        self.goal_id = 0
        self.curr_goal = self.goals[self.goal_id]
    
    def update_plan(self, robot_pos, stay_still):
        if torch.norm(robot_pos - self.curr_goal) < 0.1:
            self.goal_id += 1
            if self.goal_id >= self.goals.size(0):
                self.goal_id = 0
            self.curr_goal = self.goals[self.goal_id]

class PLANNER_AIF(PLANNER_SIMPLE):
    def __init__(self) -> None:
        PLANNER_SIMPLE.__init__(self)
        # Define the required mdp structures 
        mdp_isAt = isaac_state_action_templates.MDPIsAt()
        mdp_battery = isaac_int_req_templates.MDPBattery()  
        # Define ai agent with related mdp structure to reason about
        self.ai_agent_task = [ai_agent.AiAgent(mdp_isAt), ai_agent.AiAgent(mdp_battery)]
        # Set the preference for the battery 
        self.ai_agent_task[0].set_preferences(np.array([[1.], [0]]))
        self.battery_factor = 1.1
        self.battery_level = 100
        self.nav_goal = torch.tensor([3, -3], device="cuda:0")
    
    # Reset
    def reset_plan(self):
        self.task = "None"
        self.curr_goal = "None"
        self.battery_level = 100
        self.nav_goal = torch.tensor([3, -3], device="cuda:0")

    # Check the distance
    def robot_close_to(self, robot_pos, obj_pos):
        return torch.norm(robot_pos - obj_pos) < 0.5

    # Battery simulation
    def battery_sim(self, robot_pos, stay_still):
        obs_task = self.get_task_motion_obs(robot_pos)
        if not stay_still and obs_task:
            if self.robot_close_to(robot_pos, env_conf.docking_station_loc):
                self.battery_level += self.battery_factor
            else:
                self.battery_level -= self.battery_factor
            self.battery_level = max(0, self.battery_level)
            self.battery_level = min(100, self.battery_level)

        # save battery value to a csv file
        file_path = path_utils.get_plot_path() +'/data_battery.csv'
        np.savetxt(file_path, [self.battery_level], fmt='%.1f')

    # Battery observation
    def get_battery_obs(self, robot_pos):
        # Estimate the battery level for task
        dist_battery_factor = 40 / (3 * np.sqrt(2))
        if self.battery_level > 60 :
            battery_enough_for_task = torch.norm(robot_pos - self.nav_goal) * dist_battery_factor / (self.battery_level - 60) < 1
            battery_enough_for_task = battery_enough_for_task.item()
        else:
            battery_enough_for_task = False
        # print('enough:', battery_enough_for_task)
        if battery_enough_for_task:
            if self.battery_level > 80: 
                obs_battery = 0  # Battery is ok
            elif self.battery_level > 60:
                obs_battery = 1  # Battery is low
            else:
                obs_battery = 2  # Battery is critical
        else:
            obs_battery = 2      # Battery is critical
        return obs_battery
    
    # Task motion observation
    def get_task_motion_obs(self, robot_pos):
        if self.robot_close_to(robot_pos, self.nav_goal):
            obs_task = 0     # at_goal
        else:
            obs_task = 1     # not_at_goal
        return obs_task

    # Upadte the task planner
    def update_plan(self, robot_pos, stay_still):
        self.battery_sim(robot_pos, stay_still)
        obs_battery = self.get_battery_obs(robot_pos)
        obs_task = self.get_task_motion_obs(robot_pos)
        obs = [obs_task, obs_battery]
        outcome, curr_action = adaptive_action_selection.adapt_act_sel(self.ai_agent_task, obs)
        print('Measured battery level:', self.battery_level)
        print('Status:', outcome)
        print('Current action:', curr_action)

        if curr_action == 'go_recharge':
            self.task = 'go_recharge'
            self.curr_goal = env_conf.docking_station_loc
        elif curr_action == "move_to":
            self.task = "navigation"
            self.curr_goal = self.nav_goal
        elif curr_action == "slow_down":
            if self.robot_close_to(robot_pos, env_conf.docking_station_loc):
                self.task = "None"
                self.curr_goal = "None"
            else:
                self.task = "navigation"
                self.curr_goal = self.nav_goal