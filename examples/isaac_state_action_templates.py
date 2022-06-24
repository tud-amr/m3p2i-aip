
# This is a module which contains the templates for the classes to define MDP problems to feed to active inference
# This is the pool of actions and states that we can manipulate
import numpy as np

class MDPIsAt:
    def __init__(self): 
        self.state_name = 'isAt'                            # This is the general name the class refers to
        self.state_names = ['at_goal', 'not_at_goal']                 # These are the names a certain battery state can have
        self.action_names = ['idle', 'move_to']    # These are the names of the actions for internal needs

        self.V = np.array([0, 1])  # Allowable policies, it indicates policies of depth 1
        self.B = np.zeros((2, 2, 2))  # Allowable actions initiation
        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(2)  # Idle action
        self.B[:, :, 1] = np.array([[1, 1],  # move(loc): a_moveBase makes isAt true
                                    [0, 0]])

        # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['none'], ['none']]    # No preconditions needed for Idle and move_to                       

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

class MDPIsHolding:
    def __init__(self):
        self.state_name = 'isHolding'                            # This is the general name the class refers to
        self.state_names = ['holding_obj', 'not_holding_obj']                 # These are the names a certain battery state can have
        self.action_names = ['idle', 'pick', 'place_somewhere']    # These are the names of the actions for internal needs

        self.V = np.array([0, 1, 2])  # Allowable policies, it indicates policies of depth 1
        self.B = np.zeros((2, 2, 3))  # Allowable actions initiation
        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(2)  # Idle action
        self.B[:, :, 1] = np.array([[1, 1],  # Pick action
                                    [0, 0]])
        self.B[:, :, 2] = np.array([[0, 0],  # Place_somewhere action
                                    [1, 1]])
        # Preconditions of the actions above
        self.preconditions = [['none'], ['not_holding_obj', 'reachable', 'visible'], ['none']] # [Idle precondition], [pick preconditions], [place precondition]  

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
        self.E = np.array([[1.01], [1], [1]])
        # Learning rate for initial state update
        # -----------------------------------------------------------
        self.kappa_d = 1

class MDPIsReachable:
    def __init__(self): 
        self.state_name = 'isReachable'                            # This is the general name the class refers to
        self.state_names = ['reachable', 'not_reachable']                 # These are the names a certain battery state can have
        self.action_names = ['idle', 'move_to_reach']    # These are the names of the actions for internal needs

        self.V = np.array([0, 1])  # Allowable policies, it indicates policies of depth 1
        self.B = np.zeros((2, 2, 2))  # Allowable actions initiation
        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(2)             # Idle action
        self.B[:, :, 1] = np.array([[1, 1],     # move_to_reach
                                    [0, 0]])

        # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['none'], ['none']]    # No preconditions needed for Idle and move_to_reach                       
           

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

class MDPIsVisible:
    def __init__(self): 
        self.state_name = 'isVisible'                            # This is the general name the class refers to
        self.state_names = ['visible', 'not_visible']                 # These are the names a certain battery state can have
        self.action_names = ['idle', 'look_around']    # These are the names of the actions for internal needs

        self.V = np.array([0, 1])  # Allowable policies, it indicates policies of depth 1
        self.B = np.zeros((2, 2, 2))  # Allowable actions initiation
        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(2)             # Idle action
        self.B[:, :, 1] = np.array([[1, 1],     # look_around
                                    [0, 0]])

        # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['none'], ['none']]    # No preconditions needed for Idle and look_around                       
           

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

class MDPIsPlacedAt:
    def __init__(self):
        self.state_name = 'isInBasket'                                                  # This is the general name the class refers to
        self.state_names = ['placed_in_basket', 'not_placed_in_basket']                 # These are the names a certain battery state can have
        self.action_names = ['idle', 'place_in_basket']    # These are the names of the actions for internal needs

        self.V = np.array([0, 1])       # Allowable policies, it indicates policies of depth 1
        self.B = np.zeros((2, 2, 2))    # Allowable actions initiation (idle and place_in_basket)
        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(2)             # Idle action
        self.B[:, :, 1] = np.array([[1, 1],     # Action Place_in_basket
                                    [0, 0]])
        # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['none'], ['holding_obj']]    # [idle precondition], [place_in_backet_obj]                  


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
        self.E = np.array([[1.01], [1]])
        # Learning rate for initial state update
        # -----------------------------------------------------------
        self.kappa_d = 1