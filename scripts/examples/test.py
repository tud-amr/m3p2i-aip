#!/usr/bin/env python3

# Simple example to create an AI agent which selects actions to satisfy task needs

import numpy as np
import sys
sys.path.append('../')
from active_inference import ai_agent, state_action_templates, adaptive_action_selection

## Initialization
# ----------------- 
# Define the required mdp structures from the templates
mdp_isAt = state_action_templates.MDPIsAt() 
mdp_isHolding = state_action_templates.MDPIsHolding() 
mdp_isReachable = state_action_templates.MDPIsReachable() 
mdp_isPlacedAt = state_action_templates.MDPIsPlacedAt() 
mdp_isVisible = state_action_templates.MDPIsVisible() 

# Agent with following states [isAt, isHolding, isReachable, isPlacedAt, isVisible]
ai_agent_task = [ai_agent.AiAgent(mdp_isAt), ai_agent.AiAgent(mdp_isHolding), ai_agent.AiAgent(mdp_isReachable), ai_agent.AiAgent(mdp_isPlacedAt), ai_agent.AiAgent(mdp_isVisible)]

# Loop for the execution of the task, ideally this will be given by the tick of a BT
for i in range(20):
    # Set preference for a particular state to be achieved. Follow indexing in ai_agent_task [isAt, isHolding, isReachable, isPlacedAt, isVisible]
    ai_agent_task[3].set_preferences(np.array([[1.], [0.]]))
    #ai_agent_task[1].set_preferences(np.array([[2.], [0.]]))
    ai_agent_task[1]._mdp.D = (np.array([[0], [1]]))

    # Get the parameters of the object to be holding to define a suitable observation
    # most likely from the parameter server (TODO add the parameters from the BT)

    # Set the observation from the current readings, (TODO) the logic of the observations need to be specified for the task and the parameters passed by the BT
    # in terms of products. When an observation is unavailable set it to 'null'
    if i < 5:
        # should reach
        is_holding = False
        is_reachable = False
        close_to_pre_place = False
    elif i < 10:
        # should pick
        is_holding = False
        is_reachable = True
        close_to_pre_place = False
    elif i< 15:
        # should pick
        is_holding = True
        is_reachable = True
        close_to_pre_place = False
    elif i < 20:
        # should place
        is_holding = True
        is_reachable = True
        close_to_pre_place = True
    # Translate to integer observations, do not worry about the negatives, 0 means true
    obs = ['null', int(not is_holding), int(not is_reachable), int(not close_to_pre_place), 0]
    outcome, curr_acti = adaptive_action_selection.adapt_act_sel(ai_agent_task, obs)
    print(i, curr_acti)
    # print(ai_agent_task[3]._mdp.D)