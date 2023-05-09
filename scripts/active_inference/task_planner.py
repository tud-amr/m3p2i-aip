import torch
import numpy as np
from utils import env_conf
from active_inference import ai_agent, adaptive_action_selection
from active_inference import isaac_int_req_templates, isaac_state_action_templates
import sys
import time
import os
sys.path.append('../')
from utils import path_utils, skill_utils

class PLANNER_SIMPLE:
    def __init__(self, task, goal) -> None:
        self.task = task
        self.curr_goal = goal if torch.is_tensor(goal) else torch.tensor(goal, device="cuda:0")

    def update_params(self, params):
        if self.task == "pull":
            params.suction_active = True
        else:
            params.suction_active = False
        return params

    def update_plan(self, robot_pos, stay_still):
        pass
    
    def reset_plan(self):    
        pass

    def check_task_success(self, robot_pos, block_pos):
        if self.task in ['navigation', 'go_recharge']:
            task_success = torch.norm(robot_pos - self.curr_goal) < 0.1
        elif self.task in ['push', 'pull', 'hybrid']:
            task_success = torch.norm(block_pos - self.curr_goal) < 0.15
        else:
            task_success = False
        return task_success

class PLANNER_PICK(PLANNER_SIMPLE):
    def __init__(self, task, goal) -> None:
        PLANNER_SIMPLE.__init__(self, task, goal)
        self.initial_goal = self.curr_goal.clone()

    def update_plan(self, cube_state, cube_goal, ee_goal):
        if self.task == 'pick':
            cube_goal = cube_goal.clone()
            cube_state = cube_state.clone()
            ee_goal = ee_goal.clone()
            cube_goal[2] += 0.052
            self.curr_goal = cube_goal
            norm0 = torch.linalg.norm(cube_goal[:2] - cube_state[:2])
            norm1 = torch.linalg.norm(cube_goal[:3] - cube_state[:3])
            norm2 = torch.linalg.norm(cube_goal[3:] - cube_state[3:])
            print('norm0', norm0)
            print('norm1', norm1)
            print('norm2', norm2)
            a = skill_utils.quaternion_rotation_matrix(cube_goal[3:].view(-1,4))
            dist_cost = torch.linalg.norm(cube_goal[:3] - cube_state[:3])
            ori_cost = skill_utils.get_quaternions_ori_cost(cube_goal[3:].view(-1,4), cube_state[3:].view(-1,4))
            print('ori', ori_cost[0])
            # print('state', cube_state - cube_goal)
            if dist_cost + ori_cost < 0.01:
                self.task = 'place'
                ee_goal[2] += 0.2
                self.curr_goal = ee_goal

    def reset_plan(self):    
        self.task = 'pick'
        self.curr_goal = self.initial_goal

class PLANNER_PATROLLING(PLANNER_SIMPLE):
    def __init__(self, goals) -> None:
        self.task = "navigation"
        self.goals = torch.tensor(goals, device="cuda:0")
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
        PLANNER_SIMPLE.__init__(self, "navigation", [0, 0])
        # Define the required mdp structures 
        mdp_isAt = isaac_state_action_templates.MDPIsAt()
        mdp_battery = isaac_int_req_templates.MDPBatteryTask()  
        # Define ai agent with related mdp structure to reason about
        self.ai_agent_task = [ai_agent.AiAgent(mdp_isAt), ai_agent.AiAgent(mdp_battery)]
        # Set the preference for the battery 
        self.ai_agent_task[0].set_preferences(np.array([[1.], [0]]))
        self.battery_factor = 1  # <=0.8 0.9 >=1.0
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
            battery_enough_for_task = torch.norm(robot_pos - self.nav_goal) * dist_battery_factor * self.battery_factor / (self.battery_level - 60) < 1
            battery_enough_for_task = battery_enough_for_task.item()
        else:
            battery_enough_for_task = False
        obs_battery = int(not battery_enough_for_task)
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
        print('Measured battery level:', format(self.battery_level, '.2f'))
        print('Status:', outcome)
        print('Current action:', curr_action)

        if curr_action == 'go_recharge':
            self.task = 'go_recharge'
            self.curr_goal = env_conf.docking_station_loc
        elif curr_action == "move_to":
            self.task = "navigation"
            self.curr_goal = self.nav_goal