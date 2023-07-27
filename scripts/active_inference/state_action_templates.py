import numpy as np

class MDPIsAtPlaceLoc:
    def __init__(self): 
        self.state_name = 'isAtPlaceLoc'                               
        self.state_names = ['at_place_loc', 'not_place_loc']          
        self.action_names = ['idle', 'move_to_place']                 

        self.V = np.array([0, 1])          
        self.B = np.zeros((2, 2, 2))       

        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(2)  # Idle action
        self.B[:, :, 1] = np.array([[1, 1],         # move_to
                                    [0, 0]])

        # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['none'], ['holding_obj']]    # No preconditions needed for Idle and move_to                       

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

class MDPIsReachable:
    def __init__(self): 
        self.state_name = 'isReachable'                            
        self.state_names = ['reachable', 'not_reachable']                
        self.action_names = ['idle', 'reach']    

        self.V = np.array([0, 1])  # Allowable policies, it indicates policies of depth 1
        self.B = np.zeros((2, 2, 2))  # Allowable actions initiation
        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(2)             # Idle action
        self.B[:, :, 1] = np.array([[1, 1],     # reach
                                    [0, 0]])

        # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['none'], ['none']]    # No preconditions needed for Idle and reach                       
           

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
        self.state_name = 'isHolding'                            
        self.state_names = ['holding_obj', 'not_holding_obj']               
        self.action_names = ['idle', 'pick']  

        self.V = np.array([0, 1])  
        self.B = np.zeros((2, 2, 2))
        
        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(2)  # Idle action
        self.B[:, :, 1] = np.array([[1, 1],  # Pick action
                                    [0, 0]])
        # Preconditions of the actions above
        self.preconditions = [['none'], ['reachable'], ['none']]

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

class MDPIsPlacedOn:
    def __init__(self):
        self.state_name = 'isOnTop'                                                  
        self.state_names = ['placed_on', 'not_placed_on']               
        self.action_names = ['idle', 'place']   

        self.V = np.array([0, 1])       
        self.B = np.zeros((2, 2, 2))  
        
        # Transition matrices
        # ----------------------------------------------------------
        self.B[:, :, 0] = np.eye(2)             # Idle action
        self.B[:, :, 1] = np.array([[1, 1],     # Action Place_on
                                    [0, 0]])
        # Preconditions of the actions above
        # ----------------------------------------------------------
        self.preconditions = [['none'], ['holding_obj', 'at_place_loc']]    # [idle precondition], [place_in_backet_obj]                  


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