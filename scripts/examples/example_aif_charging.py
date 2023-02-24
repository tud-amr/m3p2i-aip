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
mdp_battery = isaac_int_req_templates.MDPBatteryNew() 
mdp_charging = isaac_int_req_templates.MDPCharging()

# Agent with following states [isAt, batteryState], see templates
ai_agent_task = [ai_agent.AiAgent(mdp_isAt), ai_agent.AiAgent(mdp_battery), ai_agent.AiAgent(mdp_charging)]
ai_agent_task[0].set_preferences(np.array([[1.], [0]]))
# ai_agent_task[1].set_preferences(np.array([[1.], [0], [0]])) 
# ai_agent_task[2].set_preferences(np.array([[1.], [0]])) 

# Loop for the execution of the task, ideally this will be given by the tick of a BT
for i in range(35):
    # Set the observation from the current readings, the logic of the observations need to be specified for the task. 
    # When an observation is unavailable set it to 'null'

    '''
    *******************************************************
    Case 1: The battery is enough for completing the task
    *******************************************************
    '''
    # if i < 5:      
    #     obs = [1, 0, 1] # not_at_goal, battery_ok, not_charging -> move_to
    # elif i < 10:   
    #     obs = [1, 1, 1] # not_at_goal, battery_low, not_charging -> slow down
    # else:          
    #     obs = [0, 2, 1] # at_goal, battery_critical, not_charging -> idle_success
    # # print('pref', ai_agent_task[0]._mdp.C)

    '''
    *************************************************************
    Case 2: The battery is not enough for completing the task,
    so it will go to recharge halfway and complete the task later
    *************************************************************
    '''
    if i < 5:     
        obs = [1, 0, 1] # not_at_goal, battery_ok, not_charging -> move_to
    elif i < 10:
        obs = [1, 1, 1] # not_at_goal, battery_low, not_charging -> slow down
    elif i < 15:
        ai_agent_task[1].set_preferences(np.array([[0.], [1], [0]])) # favors battery_low
        obs = [1, 2, 1] # not_at_goal, battery_critical, not_charging -> go_recharge
    elif i < 20:
        ai_agent_task[2].set_preferences(np.array([[1.], [0]])) # favors is_charging
        obs = [1, 2, 0] # not_at_goal, battery_critical, is_charging -> idle
    elif i < 25:
        ai_agent_task[2].set_preferences(np.array([[1.], [0]])) # favors is_charging
        obs = [1, 1, 0] # not_at_goal, battery_low, is_charging -> idle
    elif i < 30:
        ai_agent_task[2].set_preferences(np.array([[0.], [1]])) # favors move_to
        obs = [1, 0, 0] # not_at_goal, battery_ok, is_charging -> move_to
    else:
        obs = [0, 1, 1] # at_goal, battery_low, not_charging -> idle_success


    # To test parallel adaptive action selection swap commented lines below
    outcome, curr_acti = adaptive_action_selection.adapt_act_sel(ai_agent_task, obs)
    #outcome, curr_acti = parallel_action_selection.par_act_sel(ai_agent_task, obs)

    print('Round:', i)
    # print('Status:', outcome)
    print('Current action(s):', curr_acti)