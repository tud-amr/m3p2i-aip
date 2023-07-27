#!/usr/bin/env python3

# Simple example to create an AI agent which selects actions to satisfy task needs

import numpy as np
import sys
sys.path.append('../')
from active_inference import ai_agent, state_action_templates, adaptive_action_selection

mdp_isAtPlaceLoc = state_action_templates.MDPIsAtPlaceLoc() 
mdp_isHolding = state_action_templates.MDPIsHolding() 
mdp_isPlacedOn = state_action_templates.MDPIsPlacedOn() 
mdp_isReachable = state_action_templates.MDPIsReachable() 

# Agent with following states [isAtPlaceLoc, isHolding, isReachable, isPlacedOn], see templates
ai_agent_task = [ai_agent.AiAgent(mdp_isAtPlaceLoc), ai_agent.AiAgent(mdp_isHolding), ai_agent.AiAgent(mdp_isReachable), ai_agent.AiAgent(mdp_isPlacedOn)]
# Define the task for an agent by setting the preferences
ai_agent_task[3].set_preferences(np.array([[1.], [0.]]))

# Loop for the execution of the task, ideally this will be given by the tick of a BT
for i in range(32):
    # Set the observation from the current readings, the logic of the observations need to be specified for the task. 
    # When an observation is unavailable set it to 'null'
    
    # Test
    if i < 3:
        obs = [1, 1, 1, 1]         # not at place location, not picked, not, reachable, not placed
    if i >= 3 and i < 5:
        obs = [1, 1, 0, 1]         # not at place location, not picked, reachable, not placed
    if i>= 5 and i < 8:        
        obs = [1, 0, 0, 1]         # not at place location, picked, reachable, not placed
    if i>= 8 and i < 10:   # simulate losing it while carrying    
        obs = [1, 1, 1, 1]         # not at place location, not picked, not,reachable, not placed
    if i>= 10 and i < 12:   
        obs = [1, 1, 0, 1]         # not at place location, not picked, reachable, not placed
    if i>= 12 and i < 14:   
        obs = [1, 0, 0, 1]         # not at place location, picked, reachable, not placed
    if i>= 14 and i < 15:   
        obs = [0, 0, 0, 1]         # at place location, picked, not placed
    if i>= 15 and i < 20:
        obs = [0, 0, 0, 0]         # success
    if i>= 20 and i < 23:   # simulate losing it after placing correctly
        obs = [1, 1, 1, 1]         # not at place location, not picked, not reachable, not placed
    if i>= 23 and i < 25:        
        obs = [1, 1, 0, 1]         # not at place location, not picked, reachable not placed
    if i>= 25 and i < 28:   
        obs = [1, 0, 0, 1]         # not at place location, picked, reachable, not placed
    if i>= 28 and i < 30:
        obs = [0, 0, 0, 1]         # not at place location, picked, reachable, not placed
    if i>= 30 and i < 32:
        obs = [0, 0, 0, 0]         # success

    outcome, curr_acti = adaptive_action_selection.adapt_act_sel(ai_agent_task, obs)
   
    # print('Status:', outcome)
    print('Current action(s):', curr_acti)