# This is a module which contains the templates for the classes to define MDP problems to feed to active inference
# It contains the internal requirements of the system and how the agent can act to mantain the preferred levels

import numpy as np

class MDPBatteryTask:
    def __init__(self): 
        self.state_name = 'batteryStateForTask'                 # This is the general name the class refers to
        self.state_names = ['battery_ok', 'battery_not_enough'] # These are the names a certain battery state can have
        self.action_names = ['move_to', 'go_recharge']          # These are the names of the actions for internal needs

        self.V = np.array([0, 1])                               # Allowable policies, it indicates policies of depth 1
        self.B = np.zeros((2, 2, 2))                            # Allowable actions initiation

        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.array([[1, 0],  # move(loc)
                                    [0, 1]])
        self.B[:, :, 1] = np.array([[1, 1],  # go_recharge
                                    [0, 0]])

        # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['battery_ok'], ['battery_not_enough']]    # No preconditions needed for Idle and move_to                       

        # Likelihood matrix matrices
        # ----------------------------------------------------------
        self.A = np.eye(2)  # Identity mapping
        # Prior preferences, initially set to zero, so no preference
        # -----------------------------------------------------------
        self.C = np.array([[0.], [0.]])
        # Belief about initial state, D
        # -----------------------------------------------------------
        self.D = np.array([[0.5], [0.5]])

        # Preference about actions, idle is slightly preferred
        # -----------------------------------------------------------
        self.E = np.array([[1.01], [1]])
        # Learning rate for initial state update
        # -----------------------------------------------------------
        self.kappa_d = 1