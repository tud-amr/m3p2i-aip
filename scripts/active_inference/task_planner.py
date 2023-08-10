import torch
import numpy as np
from utils import env_conf
from active_inference import ai_agent, adaptive_action_selection
from active_inference import isaac_int_req_templates, isaac_state_action_templates, state_action_templates
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

    def check_task_success(self, robot_pos, block_state):
        if self.task in ['navigation', 'go_recharge']:
            task_success = torch.norm(robot_pos - self.curr_goal) < 0.1
        elif self.task in ['push', 'pull', 'hybrid']:
            pos_dist = torch.norm(block_state[:2] - self.curr_goal)
            goal_quat = torch.tensor([0, 0, 0, 1], device="cuda:0").view(1, 4)
            ori_dist = skill_utils.get_general_ori_cube2goal(block_state[3:7].view(1, 4), goal_quat)
            # print('pos', pos_dist)
            # print('ori', ori_dist)
            task_success = pos_dist <= 0.15 and ori_dist <= 0.1
        else:
            task_success = False
        return task_success

class PLANNER_AIF_PANDA_REAL(PLANNER_SIMPLE):
    def __init__(self) -> None:
        PLANNER_SIMPLE.__init__(self, "idle", [0, 0, 0, 0, 0, 0, 0])
        # Define the required mdp structures from the templates
        mdp_isCubeAtReal = isaac_state_action_templates.MDPIsCubeAtReal()

        # Agent with following states [isCubeAt]
        self.ai_agent_task = [ai_agent.AiAgent(mdp_isCubeAtReal)]
        self.obs = 0
        self.prev_ee_state = torch.zeros(7, device="cuda:0")
        self.pick_always = False
        self.place_forever = False
        self.ee_goal_once = 0

    def get_obs(self, cube_state, cube_goal, ee_state):
        cube_height_diff = torch.linalg.norm(cube_state[2] - cube_goal[2])
        reach_cost = torch.linalg.norm(ee_state[:3] - cube_state[:3])
        dist_cost = torch.linalg.norm(self.curr_goal[:3] - cube_state[:3]) # self.curr_goal !!
        ori_cost = skill_utils.get_general_ori_cube2goal(self.curr_goal[3:].view(-1,4), cube_state[3:].view(-1,4))
        print('cost reach hh', reach_cost)
        # print()
        print('dis', dist_cost)
        print('ori', ori_cost[0])
        if (dist_cost < 0.015 and ori_cost < 0.01) or self.place_forever:
            self.obs = 2
            if self.ee_goal_once <1:
                self.ee_goal = ee_state.clone()
                self.ee_goal[2] += 0.2
            self.ee_goal_once += 1
            self.place_forever = True
            self.ai_agent_task[0].set_preferences(np.array([[1.], [0], [0], [0]]))
            # print('end hh', self.end_goal[:3])
            # print(torch.linalg.norm(self.end_goal[:3]-ee_state[:3]))
            if torch.linalg.norm(self.end_goal[:3]-ee_state[:3]) < 0.06:
                self.obs = 3
        elif reach_cost < 0.012 or self.pick_always: # not too picky
            self.obs = 1
            self.pick_always = True
            self.ai_agent_task[0].set_preferences(np.array([[1], [0], [0], [0]]))
        elif cube_height_diff < 0.01 and not self.pick_always: # reach
            self.obs = 0
            self.ai_agent_task[0].set_preferences(np.array([[0], [1], [0], [0]]))

    def update_plan(self, cube_state, cube_goal, ee_state):
        self.get_obs(cube_state, cube_goal, ee_state)
        # outcome, curr_action = adaptive_action_selection.adapt_act_sel(self.ai_agent_task, [self.obs])
        # print('Status:', outcome)
        # print('Current action:', curr_action)

        if self.obs == 0:
            curr_action = 'reach'
        elif self.obs == 1:
            curr_action = 'pick'
        else:
            curr_action = 'place'
        # elif self.obs == 3:
        #     curr_action = 'idle'
        
        self.task = curr_action
        if curr_action == 'reach':
            self.curr_goal = torch.zeros(7, device='cuda:0')
        elif curr_action == 'pick':
            self.curr_goal = cube_goal.clone()
            self.curr_goal[2] += 0.085
            self.end_goal = self.curr_goal.clone()
            self.end_goal[2] += 0.1
        elif curr_action == "place":
            # self.ai_agent_task[0].set_preferences(np.array([[1], [0], [0]]))
            self.curr_goal = self.ee_goal
        elif curr_action == "idle":
            self.curr_goal = self.ee_goal
    
    def check_task_success(self, ee_state):
        flag = False
        if self.task == 'place' and torch.linalg.norm(ee_state - self.prev_ee_state) < 0.001:
            flag = True
        self.prev_ee_state = ee_state.clone()
        return flag

class PLANNER_AIF_PANDA_REACTIVE(PLANNER_SIMPLE):
    def __init__(self) -> None:
        PLANNER_SIMPLE.__init__(self, "idle", [0, 0, 0, 0, 0, 0, 0])
        # Define the required mdp structures from the templates
        mdp_isAtPlaceLoc = state_action_templates.MDPIsAtPlaceLoc() 
        mdp_isHolding = state_action_templates.MDPIsHolding() 
        mdp_isPlacedOn = state_action_templates.MDPIsPlacedOn() 
        mdp_isReachable = state_action_templates.MDPIsReachable() 

        # Agent with following states [isAtPlaceLoc, isHolding, isReachable, isPlacedOn], see templates
        self.ai_agent_task = [ai_agent.AiAgent(mdp_isAtPlaceLoc), 
                              ai_agent.AiAgent(mdp_isHolding), 
                              ai_agent.AiAgent(mdp_isReachable), 
                              ai_agent.AiAgent(mdp_isPlacedOn)]
        # Define the task for an agent by setting the preferences
        self.ai_agent_task[3].set_preferences(np.array([[1.], [0.]]))

    def get_obs(self, cube_state, cube_goal, ee_state, gripper_dist):
        cube_height_diff = torch.linalg.norm(cube_state[2] - cube_goal[2])
        reach_cost = torch.linalg.norm(ee_state[:3] - cube_state[:3])
        self.pre_place_goal = cube_goal.clone()
        self.pre_place_goal[2] += 0.085
        pre_place_dist_cost = torch.linalg.norm(self.pre_place_goal[:3] - cube_state[:3])
        # uni-modal
        # pre_place_ori_cost = skill_utils.get_general_ori_cube2goal(self.pre_place_goal[3:].view(-1,4), cube_state[3:].view(-1,4))
        # multi-modal
        pre_place_ori_cost = skill_utils.get_general_ori_cube2goal(ee_state[3:].view(-1, 4), self.pre_place_goal[3:].view(-1,4))
        final_dist_cost = torch.linalg.norm(cube_goal[:2] - cube_state[:2])
        final_ori_cost = skill_utils.get_general_ori_cube2goal(cube_goal[3:].view(-1,4), cube_state[3:].view(-1,4))
        print('reach cost', reach_cost)
        print('gripper dis', gripper_dist)
        print('pre place dis', pre_place_dist_cost)
        print('pre place ori', pre_place_ori_cost[0])
        # print('final dis', final_dist_cost)
        # print('final ori', final_ori_cost[0])

        # Get observatios, 0 means True, 1 means False
        is_holding = 0 if gripper_dist < 0.065 and gripper_dist > 0.058 and reach_cost < 0.03 else 1 #!!
        is_close_to_pre_place = 0 if pre_place_dist_cost < 0.01 and pre_place_ori_cost < 0.01 else 1
        self.is_success = 0 if cube_height_diff < 0.065 and final_dist_cost < 0.018 and final_ori_cost < 0.01 else 1 # and final_dist_cost < 0.01
        is_reachable = 0 if not is_holding or not is_close_to_pre_place or not self.is_success or reach_cost < 0.012 else 1
        print('cube height', cube_height_diff)
        dic = {'is_close_to_pre':is_close_to_pre_place, 
               'is_holding':is_holding, 
               'is_reachable':is_reachable,
               'is_success':self.is_success}
        print(dic)

        self.obs = [is_close_to_pre_place, 
                    is_holding, 
                    is_reachable, 
                    self.is_success]

    def update_plan(self, cube_state, cube_goal, ee_state, gripper_dist):
        self.get_obs(cube_state, cube_goal, ee_state, gripper_dist)
        outcome, curr_action = adaptive_action_selection.adapt_act_sel(self.ai_agent_task, self.obs)
        # print('Status:', outcome)
        print('Current action:', curr_action)
        
        self.task = curr_action
        if curr_action == 'move_to_place':
            self.curr_goal = self.pre_place_goal
        else:
            self.curr_goal = torch.zeros(7, device='cuda:0')
    
    def check_task_success(self, ee_state):
        return not self.is_success