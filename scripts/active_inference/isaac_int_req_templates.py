# This is a module which contains the templates for the classes to define MDP problems to feed to active inference
# It contains the internal requirements of the system and how the agent can act to mantain the preferred levels

import numpy as np

class MDPBattery():
    def __init__(self): 
        self.state_name = 'batteryState'                            # This is the general name the class refers to
        self.state_names = ['ok', 'low', 'critcal']                 # These are the names a certain battery state can have
        self.action_names = ['go_normal', 'slow_down', 'go_recharge']    # These are the names of the actions for internal needs

        self.V = np.array([0, 1, 2])            # Allowable policies, it indicates policies of depth 1 (there are 3 policies)
        self.B = np.zeros((3, 3, 3))            # Allowable actions initiation
        
        # Transition matrices corecponding to the actions
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(3)                 # go normal     (remove velocity cost from cost function)
        self.B[:, :, 1] = np.array([[1, 1, 0],      # slow down     (penalize velocities)
                                    [0, 0, 1],
                                    [0, 0, 0]])    
        self.B[:, :, 2] = np.array([[1, 1, 1],      # go recharging (penalize velocities and trigger awareness at high level)
                                    [0, 0, 0],
                                    [0, 0, 0]]) 
        # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['none'], ['none'], ['none']]    # No preconditions needed for actions
        
        # Likelihood matrix matrices
        # ----------------------------------------------------------
        self.A = np.eye(3)  # Identity mapping
        # Prior preferences. Initially none
        # -----------------------------------------------------------
        self.C = np.array([[0], [0], [0]])
        # Initial guess about the states d, all equally possible, this is updated over time
        # -----------------------------------------------------------
        self.D = np.array([[0.5], [0.5], [0.5]])
        # Preference about actions. The agent prefers to move normally, slightly less to go slow, and even less to go recharge
        # -----------------------------------------------------------
        self.E = np.array([[1.], [1.], [1.]])
        # Learning rate for state update
        # -----------------------------------------------------------
        self.kappa_d = 0.8