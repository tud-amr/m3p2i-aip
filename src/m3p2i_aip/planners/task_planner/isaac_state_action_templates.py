
# This is a module which contains the templates for the classes to define MDP problems to feed to active inference
# This is the pool of actions and states that we can manipulate
import numpy as np

class MDPIsAt:
    def __init__(self): 
        self.state_name = 'isAt'                                # This is the general name the class refers to
        self.state_names = ['at_goal', 'not_at_goal']           # These are the names a certain battery state can have
        self.action_names = ['idle', 'move_to']                 # These are the names of the actions for internal needs

        self.V = np.array([0, 1])                               # Allowable policies, it indicates policies of depth 1
        self.B = np.zeros((2, 2, 2))                            # Allowable actions initiation

        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(2)  # Idle action
        self.B[:, :, 1] = np.array([[1, 1],  # move(loc): a_moveBase makes isAt true
                                    [0, 0]])

        # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['none'], ['battery_ok']]    # No preconditions needed for Idle and move_to                       

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

class MDPIsCloseTo:
    def __init__(self): 
        self.state_name = 'isCloseTo'                               
        self.state_names = ['close_to', 'not_close_to']          
        self.action_names = ['idle', 'approach_obj']                 

        self.V = np.array([0, 1])                              
        self.B = np.zeros((2, 2, 2))                           

        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(2)  # Idle action
        self.B[:, :, 1] = np.array([[1, 1],  # apprach 
                                    [0, 0]])

        # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['none'], ['none']]    # No preconditions needed                 

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

class MDPIsLocFree:
    def __init__(self):
        self.state_name = 'isLocFree'                                              
        self.state_names = ['loc_free', 'not_loc_free']                     
        self.action_names = ['idle', 'push_to_non_goal', 'pull_to_non_goal']        

        self.V = np.array([0, 1, 2]) 
        self.B = np.zeros((2, 2, 3)) 
        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(2)             # Idle action
        self.B[:, :, 1] = np.array([[1, 1],     # push_to_non__goal
                                    [0, 0]])
        self.B[:, :, 2] = np.array([[1, 1],     # pull_to_non_goal
                                    [0, 0]])

        # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['none'], ['close_to'], ['close_to']]                 

        # Likelihood matrix matrices
        # ----------------------------------------------------------
        self.A = np.eye(2)  # Identity mapping
        # Prior preferences, initially set to zero, so no preference
        # -----------------------------------------------------------
        self.C = np.array([[0.], [0.]])
        # Belief about initial state, D
        # -----------------------------------------------------------
        self.D = np.array([[0.5], [0.5]])
        # Initial guess about the states d, all equally possible, this is updated over time
        # -----------------------------------------------------------
        self.d = np.array([[0.5], [0.5]])
        # Preference about actions, idle is slightly preferred
        # -----------------------------------------------------------
        self.E = np.array([[1.01], [1], [1]])
        # Learning rate for initial state update
        # -----------------------------------------------------------
        self.kappa_d = 1

class MDPIsBlockAt:
    def __init__(self):
        self.state_name = 'isBlockAt'                                              
        self.state_names = ['block_at_loc', 'not_block_at_loc']                     
        self.action_names = ['idle', 'push_to_goal', 'pull_to_goal']        

        self.V = np.array([0, 1, 2])      
        self.B = np.zeros((2, 2, 3))    
        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(2)             # Idle action
        self.B[:, :, 1] = np.array([[1, 1],     # push_to_goal
                                    [0, 0]])
        self.B[:, :, 2] = np.array([[1, 1],     # push_to_goal
                                    [0, 0]])

        # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['none'], ['loc_free', 'close_to'], ['loc_free', 'close_to']]                 

        # Likelihood matrix matrices
        # ----------------------------------------------------------
        self.A = np.eye(2)  # Identity mapping
        # Prior preferences, initially set to zero, so no preference
        # -----------------------------------------------------------
        self.C = np.array([[0.], [0.]])
        # Belief about initial state, D
        # -----------------------------------------------------------
        self.D = np.array([[0.5], [0.5]])
        # Initial guess about the states d, all equally possible, this is updated over time
        # -----------------------------------------------------------
        self.d = np.array([[0.5], [0.5]])
        # Preference about actions, idle is slightly preferred
        # -----------------------------------------------------------
        self.E = np.array([[1.01], [1], [1]])
        # Learning rate for initial state update
        # -----------------------------------------------------------
        self.kappa_d = 1

class MDPIsCubeAt:
    def __init__(self): 
        self.state_name = 'isCubeAt'                               
        self.state_names = ['cube_at_table', 'cube_at_hand', 'cube_at_goal']          
        self.action_names = ['idle', 'pick', 'place']   

        self.V = np.array([0, 1, 2])      
        self.B = np.zeros((3, 3, 3))    
        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(3)             # Idle action
        self.B[:, :, 1] = np.array([[1, 1, 1],  # pick
                                    [0, 0, 0],
                                    [0, 0, 0]])
        self.B[:, :, 2] = np.array([[1, 1, 1],  # place
                                    [0, 0, 0],
                                    [0, 0, 0]])
        # # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['cube_at_goal'], ['cube_at_table'], ['cube_at_hand']]
        # Likelihood matrix matrices
        # ----------------------------------------------------------
        self.A = np.eye(3)  # Identity mapping
        # Prior preferences. Initially none
        # -----------------------------------------------------------
        self.C = np.array([[0], [0], [0]])
        # Initial guess about the states d, all equally possible, this is updated over time
        # -----------------------------------------------------------
        self.D = np.array([[0.5], [0.5], [0.5]])
        # Preference about actions.
        # -----------------------------------------------------------
        self.E = np.array([[1.], [1.01], [1.]])
        # Learning rate for state update
        # -----------------------------------------------------------
        self.kappa_d = 0.8

class MDPIsCubeAtReal:
    def __init__(self): 
        self.state_name = 'isCubeAt'                               
        self.state_names = ['cube_at_table', 'cube_close_to_gripper', 'cube_at_pre_place', 'cube_at_goal']          
        self.action_names = ['idle', 'reach', 'pick', 'place']   

        self.V = np.array([0, 1, 2, 3])      
        self.B = np.zeros((4, 4, 4))    
        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(4)
        self.B[:, :, 1] = np.array([[1, 1, 1, 1],  # reach
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0], 
                                    [0, 0, 0, 0]])
        self.B[:, :, 2] = np.array([[1, 1, 1, 1],  # pick
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0], 
                                    [0, 0, 0, 0]])
        self.B[:, :, 3] = np.array([[1, 1, 1, 1],  # place
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0], 
                                    [0, 0, 0, 0]])
        # # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['cube_at_goal'], ['cube_at_table'], ['cube_close_to_gripper'], ['cube_at_pre_place']]
        # Likelihood matrix matrices
        # ----------------------------------------------------------
        self.A = np.eye(4)  # Identity mapping
        # Prior preferences. Initially none
        # -----------------------------------------------------------
        self.C = np.array([[0], [0], [0], [0]])
        # Initial guess about the states d, all equally possible, this is updated over time
        # -----------------------------------------------------------
        self.D = np.array([[0.5], [0.5], [0.5], [0.5]])
        # Preference about actions.
        # -----------------------------------------------------------
        self.E = np.array([[1.], [1.01], [1.], [1]])
        # Learning rate for state update
        # -----------------------------------------------------------
        self.kappa_d = 0.8