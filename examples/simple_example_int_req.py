#!/usr/bin/env python

# Simple example to create an AI agent which selects actions to satisfy internal needs

import numpy as np
from decision_making import ai_agent, int_req_templates               


# Function to emulate gtting an observation from the environment
def get_obs_env():
    # For the battery we have 3 possible observations  ['ok', 'low', 'critcal'] = [0, 1, 2]
    obs = 1
    return obs

## Initialization
# ----------------- 
# Define the required mdp structures 
mdp_battery = int_req_templates.MDPBattery() 

# Define ai agent with related mdp structure to reason about
ai_agent_internal = ai_agent.AiAgent(mdp_battery)

## Decision making
#-------------------
# A typical sequence for decision making, ideally this should be repeated at a certain frequency

# Set the preference for the battery 
ai_agent_internal.set_preferences(np.array([[2.], [0], [0]])) # Fixed preference for battery ok, following ['ok', 'low', 'critcal'] 

for i in range(20):
    # Set the observation from the environment
    if i < 10:
        obs = 1
    if i>10 and i<15:
        obs = 2
    if i>15:
        obs = 2    
    # Compute free energy and posterior states for each policy
    F, post_s = ai_agent_internal.infer_states(obs)
    # Compute expected free-energy and posterior over policies
    G, u = ai_agent_internal.infer_policies()
    # Bayesian model averaging to get current state

    # Printouts
    print('The battery state is:',  ai_agent_internal._mdp.state_names[np.argmax(ai_agent_internal.get_current_state())])

    #print('Belief about battery state', ai_agent_internal._mdp.D)
    print('The action is:', ai_agent_internal._mdp.action_names[u])

