#!/usr/bin/env python3

# Simple example to create an AI agent which selects actions to satisfy a task

import numpy as np
import sys
sys.path.append('../')
from active_inference import ai_agent, isaac_state_action_templates, adaptive_action_selection, isaac_int_req_templates

## Initialization
# ----------------- 
# Define the required mdp structures from the templates
mdp_isAt = isaac_state_action_templates.MDPIsAt() 
mdp_battery = isaac_int_req_templates.MDPBatteryTask()

# Agent with following states [isAt, batteryState], see templates
ai_agent_task = [ai_agent.AiAgent(mdp_isAt), ai_agent.AiAgent(mdp_battery)]
ai_agent_task[0].set_preferences(np.array([[1.], [0]]))

# Loop for the execution of the task, ideally this will be given by the tick of a BT
for i in range(20):
    # Set the observation from the current readings, the logic of the observations need to be specified for the task. 
    # When an observation is unavailable set it to 'null'

    '''
    *******************************************************
    Case 1: The battery is enough for completing the task
    *******************************************************
    '''
    # if i < 5:      
    #     obs = [1, 0] # not_at_goal, battery_ok -> move_to
    # elif i < 10:   
    #     obs = [0, 0] # at_goal, battery_ok -> idle
    # # elif i < 15:          
    # #     obs = [0, 1] # at_goal, battery_not_enough -> go_recharge
    # # elif i < 20:          
    # #     obs = [0, 0] # at_goal, battery_ok -> move_to
    # # elif i < 25:   
    # #     obs = [0, 0] # at_goal, battery_ok -> idle

    '''
    *************************************************************
    Case 2: The battery is not enough for completing the task,
    so it will go to recharge directly and complete the task later
    *************************************************************
    '''
    if i < 5:     
        obs = [1, 1] # not_at_goal, battery_not_enough -> go_recharge
    elif i < 10:
        obs = [1, 0] # not_at_goal, battery_ok -> move_to
    elif i < 15:
        obs = [0, 0] # at_goal, battery_ok -> idle

    '''
    *************************************************************
    Case 3: The battery is thought to be enough at first, and not 
    enough later (meybe due to high velocity)
    *************************************************************
    '''
    # if i < 5:  
    #     obs = [1, 0] # not_at_goal, battery_ok -> move_to
    # elif i < 10:   
    #     obs = [1, 1] # not_at_goal, battery_not_enough -> go_recharge
    # elif i < 15:
    #     obs = [1, 0] # not_at_goal, battery_ok -> move_to
    # elif i < 20:
    #     obs = [0, 0] # at_goal, battery_ok -> idle

    # To test parallel adaptive action selection swap commented lines below
    outcome, curr_acti = adaptive_action_selection.adapt_act_sel(ai_agent_task, obs)
    #outcome, curr_acti = parallel_action_selection.par_act_sel(ai_agent_task, obs)

    print('Round:', i)
    # print('Status:', outcome)
    print('Current action(s):', curr_acti)