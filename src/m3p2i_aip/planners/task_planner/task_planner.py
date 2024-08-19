import torch
import numpy as np
from m3p2i_aip.utils import skill_utils
from m3p2i_aip.planners.task_planner import ai_agent, adaptive_action_selection
from m3p2i_aip.planners.task_planner import isaac_int_req_templates, isaac_state_action_templates

class PLANNER_SIMPLE:
    def __init__(self, task, goal) -> None:
        self.task = task
        self.curr_goal = goal if torch.is_tensor(goal) else torch.tensor(goal, device="cuda:0")

    def update_params(self, params):
        if self.task == "pull":
            params.suction_active = True
        else:
            params.suction_active = False
        return params

    def update_plan(self, robot_pos, stay_still):
        pass
    
    def reset_plan(self):    
        pass
    
    def check_task_success(self, sim):
        box_pos = sim.get_actor_position_by_name("box")[0, :2]
        box_ori = sim.get_actor_orientation_by_name("box")[0, :]
        # print("robot", sim.robot_pos[0, :], "box_pos", box_pos, "box_ori", box_ori)
        task_success = False
        if self.task == "navigation":
            task_success = torch.norm(sim.robot_pos[0, :] - self.curr_goal) < 0.1
        elif self.task in ['push', 'pull', 'hybrid']:
            pos_dist = torch.norm(box_pos - self.curr_goal)
            goal_quat = torch.tensor([0, 0, 0, 1], device="cuda:0").view(1, 4)
            ori_dist = skill_utils.get_general_ori_cube2goal(box_ori.view(1, 4), goal_quat)
            # print('pos', pos_dist, 'ori', ori_dist)
            task_success = pos_dist <= 0.15 # and ori_dist <= 0.1
        return task_success

class PLANNER_PICK(PLANNER_SIMPLE):
    def __init__(self, task, goal) -> None:
        PLANNER_SIMPLE.__init__(self, task, goal)
        self.initial_goal = self.curr_goal.clone()

    def update_plan(self, cube_state, cube_goal, ee_state):
        if self.task == 'pick':
            cube_goal = cube_goal.clone()
            cube_state = cube_state.clone()
            ee_goal = ee_state.clone()
            # cube_goal[2] += 0.052 # panda
            cube_goal[2] += 0.2
            self.curr_goal = cube_goal
            dist_cost = torch.linalg.norm(cube_goal[:3] - cube_state[:3])
            ori_cost = skill_utils.get_general_ori_cube2goal(cube_goal[3:].view(-1,4), cube_state[3:].view(-1,4))
            print('dis', dist_cost)
            print('ori', ori_cost[0])
            # if dist_cost + ori_cost < 0.01: # panda
            if dist_cost + ori_cost < 0.05:
                self.task = 'place'
                ee_goal[2] += 0.2
                self.curr_goal = ee_goal
    
    def check_task_success(self, ee_state):
        flag = False
        if self.task == 'place' and torch.linalg.norm(ee_state - self.prev_ee_state) < 0.001:
            flag = True
        self.prev_ee_state = ee_state.clone()
        return flag

    def reset_plan(self):    
        self.task = 'pick'
        self.curr_goal = self.initial_goal

class PLANNER_AIF_PANDA(PLANNER_SIMPLE):
    def __init__(self) -> None:
        PLANNER_SIMPLE.__init__(self, "idle", [0, 0, 0, 0, 0, 0, 0])
        # Define the required mdp structures from the templates
        mdp_isCubeAt = isaac_state_action_templates.MDPIsCubeAt()

        # Agent with following states [isCubeAt]
        self.ai_agent_task = [ai_agent.AiAgent(mdp_isCubeAt)]
        self.obs = 0
        self.prev_ee_state = torch.zeros(7, device="cuda:0")

    def get_obs(self, cube_state, cube_goal, ee_state):
        cube_height_diff = torch.linalg.norm(cube_state[2] - cube_goal[2])
        dist_cost = torch.linalg.norm(self.curr_goal[:3] - cube_state[:3]) # self.curr_goal
        ori_cost = skill_utils.get_general_ori_cube2goal(self.curr_goal[3:].view(-1,4), cube_state[3:].view(-1,4))
        print('dis', dist_cost)
        print('ori', ori_cost[0])
        if cube_height_diff < 0.001:
            self.obs = 0
            self.ai_agent_task[0].set_preferences(np.array([[0], [1], [0]]))
        elif dist_cost + ori_cost < 0.025: # 0.025 0.01
            self.obs = 1
            self.ee_goal = ee_state.clone()
            self.ee_goal[2] += 0.2
            self.ai_agent_task[0].set_preferences(np.array([[1], [0], [0]]))

    def update_plan(self, cube_state, cube_goal, ee_state):
        self.get_obs(cube_state, cube_goal, ee_state)
        outcome, curr_action = adaptive_action_selection.adapt_act_sel(self.ai_agent_task, [self.obs])
        # print('Status:', outcome)
        # print('Current action:', curr_action)

        self.task = curr_action
        if curr_action == 'pick':
            self.curr_goal = cube_goal.clone()
            self.curr_goal[2] += 0.053
        elif curr_action == "place":
            self.ai_agent_task[0].set_preferences(np.array([[1], [0], [0]]))
            self.curr_goal = self.ee_goal
    
    def check_task_success(self, ee_state):
        flag = False
        if self.task == 'place' and torch.linalg.norm(ee_state - self.prev_ee_state) < 0.002:
            flag = True
        self.prev_ee_state = ee_state.clone()
        return flag

class PLANNER_PATROLLING(PLANNER_SIMPLE):
    def __init__(self, goals) -> None:
        self.task = "navigation"
        self.goals = torch.tensor(goals, device="cuda:0")
        self.goal_id = 0
        self.curr_goal = self.goals[self.goal_id]
    
    def reset_plan(self):
        self.goal_id = 0
        self.curr_goal = self.goals[self.goal_id]
    
    def update_plan(self, robot_pos, stay_still):
        if torch.norm(robot_pos - self.curr_goal) < 0.1:
            self.goal_id += 1
            if self.goal_id >= self.goals.size(0):
                self.goal_id = 0
            self.curr_goal = self.goals[self.goal_id]

class PLANNER_AIF(PLANNER_SIMPLE):
    def __init__(self, battery_factor) -> None:
        PLANNER_SIMPLE.__init__(self, "navigation", [0, 0])
        # Define the required mdp structures 
        mdp_isAt = isaac_state_action_templates.MDPIsAt()
        mdp_battery = isaac_int_req_templates.MDPBatteryTask()  
        # Define ai agent with related mdp structure to reason about
        self.ai_agent_task = [ai_agent.AiAgent(mdp_isAt), ai_agent.AiAgent(mdp_battery)]
        # Set the preference for the battery 
        self.ai_agent_task[0].set_preferences(np.array([[1.], [0]]))
        self.battery_factor = battery_factor  # <=0.8 0.9 >=1.0
        self.battery_level = 100
        self.nav_goal = torch.tensor([3, -3], device="cuda:0")
    
    # Reset
    def reset_plan(self):
        self.task = "None"
        self.curr_goal = "None"
        self.battery_level = 100
        self.nav_goal = torch.tensor([3, -3], device="cuda:0")

    # Check the distance
    def robot_close_to(self, robot_pos, obj_pos):
        return torch.norm(robot_pos - obj_pos) < 0.5

    # Battery simulation
    def battery_sim(self, robot_pos, stay_still):
        obs_task = self.get_task_motion_obs(robot_pos)
        if not stay_still and obs_task:
            if self.robot_close_to(robot_pos, env_conf.docking_station_loc):
                self.battery_level += self.battery_factor
            else:
                self.battery_level -= self.battery_factor
            self.battery_level = max(0, self.battery_level)
            self.battery_level = min(100, self.battery_level)

        # save battery value to a csv file
        file_path = path_utils.get_plot_path() +'/data_battery.csv'
        np.savetxt(file_path, [self.battery_level], fmt='%.1f')

    # Battery observation
    def get_battery_obs(self, robot_pos):
        # Estimate the battery level for task
        dist_battery_factor = 40 / (3 * np.sqrt(2))
        if self.battery_level > 60 :
            battery_enough_for_task = torch.norm(robot_pos - self.nav_goal) * dist_battery_factor * self.battery_factor / (self.battery_level - 60) < 1
            battery_enough_for_task = battery_enough_for_task.item()
        else:
            battery_enough_for_task = False
        obs_battery = int(not battery_enough_for_task)
        return obs_battery
    
    # Task motion observation
    def get_task_motion_obs(self, robot_pos):
        if self.robot_close_to(robot_pos, self.nav_goal):
            obs_task = 0     # at_goal
        else:
            obs_task = 1     # not_at_goal
        return obs_task

    # Upadte the task planner
    def update_plan(self, robot_pos, stay_still):
        self.battery_sim(robot_pos, stay_still)
        obs_battery = self.get_battery_obs(robot_pos)
        obs_task = self.get_task_motion_obs(robot_pos)
        obs = [obs_task, obs_battery]
        outcome, curr_action = adaptive_action_selection.adapt_act_sel(self.ai_agent_task, obs)
        print('Measured battery level:', format(self.battery_level, '.2f'))
        print('Status:', outcome)
        print('Current action:', curr_action)

        if curr_action == 'go_recharge':
            self.task = 'go_recharge'
            self.curr_goal = env_conf.docking_station_loc
        elif curr_action == "move_to":
            self.task = "navigation"
            self.curr_goal = self.nav_goal

class PLANNER_AIF_PUSH(PLANNER_AIF):
    def __init__(self, battery_factor) -> None:
        super().__init__(battery_factor)
        self.block_goal = env_conf.block_goal

    # Battery observation
    def get_battery_obs(self, robot_pos, block_pos):
        # Estimate the battery level for pushing task
        dist_battery_factor = 40 / (3 * np.sqrt(2))
        if self.battery_level > 60 :
            condition_1 = torch.norm(block_pos - robot_pos) < torch.norm(block_pos- env_conf.docking_station_loc) 
            condition_2 = (torch.norm(block_pos - robot_pos) + torch.norm(block_pos - self.block_goal)) * dist_battery_factor * self.battery_factor / (self.battery_level - 60) < 1
            battery_enough_for_task = condition_1 and condition_2
            # print('condition 1', condition_1)
            print('condition 2', (torch.norm(block_pos - robot_pos) + torch.norm(block_pos - self.block_goal)) * dist_battery_factor * self.battery_factor / (self.battery_level - 60))
            battery_enough_for_task = battery_enough_for_task.item()
            if torch.norm(robot_pos - env_conf.docking_station_loc) < 0.1:
                if self.battery_level > 90:
                    battery_enough_for_task = True
                else:
                    battery_enough_for_task = False
            print('battery task', battery_enough_for_task)
        else:
            battery_enough_for_task = False
        obs_battery = int(not battery_enough_for_task)
        return obs_battery

    # Upadte the task planner
    def update_plan(self, robot_pos, block_pos, stay_still):
        self.battery_sim(robot_pos, stay_still)
        obs_battery = self.get_battery_obs(robot_pos, block_pos)
        obs_task = self.get_task_motion_obs(block_pos)
        obs = [obs_task, obs_battery]
        outcome, curr_action = adaptive_action_selection.adapt_act_sel(self.ai_agent_task, obs)
        # print('Measured battery level:', format(self.battery_level, '.2f'))
        # print('Status:', outcome)
        # print('Current action:', curr_action)

        if curr_action == 'go_recharge':
            self.task = 'go_recharge'
            self.curr_goal = env_conf.docking_station_loc
        elif curr_action == "move_to":
            self.task = "push"
            self.curr_goal = self.nav_goal