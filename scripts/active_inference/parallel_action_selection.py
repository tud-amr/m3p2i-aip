# Module for adaptive action selection

# This function computes the current applicable actions to reach a desired state. The output is a list of lists containing different plans. For a list, actions
# can be executed in parallel since do not rely to the same components (assumed one action per state is parallelizable with other actions for other states).

# Author: Corrado Pezzato, TU Delft
# Last revision: 15.11.22

import numpy as np
import copy 

def par_act_sel(agent, obs):

    some_action_found = 0
    looking_for_alternatives = 0
    curr_action_plan = []

    #  At each new iteration (or tick from a behavior tree), restore all available actions and remove high priority priors that are already satisfied
    if type(agent) == list:
        n_mdps = len(agent)
    else:
        n_mdps = 1
        agent = [agent]
        obs = [obs]
    for i in range(n_mdps):
        agent[i].reset_habits()
        for index in range(len(agent[i]._mdp.C)):  # Loop over values in the prior C
            if agent[i]._mdp.C[index] > 0 and index == obs[i]:        
                # Remove precondition pushed since it has been met
                print('removed preference state', i)
                agent[i].set_preferences(0, index)
            
    # Check if we need any action at all
    for i in range(n_mdps):
        for index in range(len(agent[i]._mdp.C)):  # Loop over values in the prior
            if agent[i]._mdp.C[index] == 0 and index == obs[i]: 
                outcome = 'success'
                curr_action_plan.append('idle_success')
                some_action_found = 1
                break

    u = [-1]*n_mdps
    current_states = ['null']*n_mdps
    
    # Instead of stopping as soon as we find a solution as in adaptive_action_selection.py, keep looking for alternativ actions after removing already found ones 
    while True and 'idle_success' not in curr_action_plan:
        for i in range(n_mdps):
            # Compute free energy and posterior states for each policy if an observation is vailable
            if obs[i] != 'null':
                if not looking_for_alternatives:
                    agent[i].infer_states(obs[i])
                # Compute expected free-energy and posterior over policies
                G, u[i] = agent[i].infer_policies()
                current_states[i] = agent[i]._mdp.state_names[np.argmax(agent[i].get_current_state())]
        # If all the actions are idle, we can return success since no action is required
        if np.max(u) == 0:
            if not looking_for_alternatives and some_action_found == 0:
                print("No action found for this situation")
                outcome = 'failure'
                break
            if some_action_found >= 1:
                break
        else:
            for i in range(n_mdps):
                # Get preconditions to be satisfied for this action if it is not idle
                if u[i] > 0:
                    prec = agent[i]._mdp.preconditions[u[i]]
                    _unmet_prec = 0
                    # Check if the preconitions are satisfied and if not add preference with high priority on respective priors 
                    for item in range(len(prec)):
                        if (prec[item] not in current_states) and (prec[item]!='none'):
                            _unmet_prec = 1
                            looking_for_alternatives = 1
                            #print('There are unmet preconditions for action', agent[i]._mdp.action_names[u[i]])
                            # Get index of missing state and push a prior on that state
                            for j in range(n_mdps):
                                if prec[item] in agent[j]._mdp.state_names:
                                    agent[j].set_preferences(2, agent[j]._mdp.state_names.index(prec[item]))  # (value, index)
                            # Inhibit current action for the inner adaptation loop since missing preconditions
                            agent[i].reset_habits(u[i])
                        # If the preconditions are met after checking we can execute the action
                    if _unmet_prec == 0:
                        agent[i].reset_habits(u[i])
                        some_action_found += 1
                        outcome = 'running'
                        curr_action_plan.append([agent[i]._mdp.action_names[u[i]], i])
    
    # Parallelize current applicable actions
    parall_plans = []
    if 'idle_success' not in curr_action_plan:
        for index in range(len(curr_action_plan)):
            parall_plans.append(list(copy.deepcopy(curr_action_plan[index])))
            for k in range(len(curr_action_plan)):
                if curr_action_plan[k][1] not in parall_plans[index]:
                    parall_plans[index].append(copy.deepcopy(curr_action_plan[k][0]))
                    parall_plans[index].append(copy.deepcopy(curr_action_plan[k][1]))

        # Remove component numbers and duplicates
        for index in range(len(curr_action_plan)):
            parall_plans[index] = [x for x in parall_plans[index] if not isinstance(x, int)]
        parall_plans = list(map(list, set(map(tuple, map(set, parall_plans)))))

    return outcome, parall_plans
