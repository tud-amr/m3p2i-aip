#!/usr/bin/env python3

# Simple example to create an AI agent which selects actions to satisfy a task

import numpy as np
import sys
sys.path.append('../')
from active_inference import ai_agent, isaac_state_action_templates, adaptive_action_selection, isaac_int_req_templates

## Initialization
# ----------------- 
# Define the required mdp structures from the templates
mdp_isCubeAtReal = isaac_state_action_templates.MDPIsCubeAtReal()

# Agent with following states [isCubeAt], see templates
ai_agent_task = [ai_agent.AiAgent(mdp_isCubeAtReal)]
# ai_agent_task[0].set_preferences(np.array([[0.], [1], [0], [0]]))

# Loop for the execution of the task, ideally this will be given by the tick of a BT
for i in range(20):
    # Set the observation from the current readings, the logic of the observations need to be specified for the task. 
    # When an observation is unavailable set it to 'null'
    '''
    *************************************************************
    Case: reach -> pick -> place->idle
    *************************************************************
    '''
    if i < 5:     
        ai_agent_task[0].set_preferences(np.array([[0.], [1], [0], [0]]))
        obs = [0] # cube_at_table -> reach
    elif i < 10:
        ai_agent_task[0].set_preferences(np.array([[0], [1], [0], [0]]))
        obs = [1] # cube_closeto_gripper -> pick
    if i < 15:     
        ai_agent_task[0].set_preferences(np.array([[0.], [1], [0], [0]]))
        obs = [0] # cube_at_table -> reach
    # elif i < 15:
    #     ai_agent_task[0].set_preferences(np.array([[1.], [0], [0], [0]]))
    #     obs = [2] # cube_at_preplace -> place
    elif i < 20:
        ai_agent_task[0].set_preferences(np.array([[0.], [0], [0], [1]]))
        obs = [3] # cube_at_goal -> idle

    # To test parallel adaptive action selection swap commented lines below
    outcome, curr_acti = adaptive_action_selection.adapt_act_sel(ai_agent_task, obs)
    #outcome, curr_acti = parallel_action_selection.par_act_sel(ai_agent_task, obs)

    print('Round:', i)
    # print('Status:', outcome)
    print('Current action(s):', curr_acti)