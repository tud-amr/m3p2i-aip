#!/usr/bin/env python

# Simple example to create an AI agent which selects actions to satisfy task needs

import numpy as np
from decision_making import ai_agent, state_action_templates               


# Function to emulate gtting an observation from the environment
def get_obs_env():
    # For the isAt state we only have two values,  ['at_goal', 'not_at_goal'] = [0, 1]. The observation will depend on the actual position of the robot in the sim
    obs = 0   # True
    return obs

## Initialization
# ----------------- 
# Define the required mdp structures 
mdp_isAt = state_action_templates.MDPIsAt() 

# Define ai agent with related mdp structure to reason about
ai_agent_task = ai_agent.AiAgent(mdp_isAt)

## Decision making
#-------------------
# A typical sequence for decision making, ideally this should be repeated at a certain frequency

# Set the preference being somewhere
ai_agent_task.set_preferences(np.array([[1.], [0]])) # Fixed preference for battery ok, following ['at_goal', 'not_at_goal']


for i in range(20):
    # Set the observation from the environment
    if i < 10:
        obs = 0
    if i>10 and i<15:
        obs = 1
    if i>15:
        obs = 1    
    # Compute free energy and posterior states for each policy
    F, post_s = ai_agent_task.infer_states(obs)
    # Compute expected free-energy and posterior over policies
    G, u = ai_agent_task.infer_policies()
    # Bayesian model averaging to get current state

    # Printouts
    print('The current state is:',  ai_agent_task._mdp.state_names[np.argmax(ai_agent_task.get_current_state())])

    #print('Belief about battery state', ai_agent_task._mdp.D)
    print('The action is:', ai_agent_task._mdp.action_names[u])

