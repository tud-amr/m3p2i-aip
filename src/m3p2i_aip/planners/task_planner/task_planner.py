import torch
import numpy as np
from m3p2i_aip.utils import skill_utils
from m3p2i_aip.planners.task_planner import ai_agent, adaptive_action_selection
from m3p2i_aip.planners.task_planner import isaac_int_req_templates, isaac_state_action_templates

class PLANNER_SIMPLE:
    def __init__(self, cfg) -> None:
        self.device = cfg.mppi.device
        self.task = cfg.task
        self.curr_goal = cfg.goal if torch.is_tensor(cfg.goal) else torch.tensor(cfg.goal, device=self.device)

    def update_plan(self, sim):
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
        elif self.task in ['push', 'pull', 'push_pull']:
            pos_dist = torch.norm(box_pos - self.curr_goal)
            goal_quat = torch.tensor([0, 0, 0, 1], device=self.device).view(1, 4)
            ori_dist = skill_utils.get_general_ori_cube2goal(box_ori.view(1, 4), goal_quat)
            # print('pos', pos_dist, 'ori', ori_dist)
            task_success = pos_dist <= 0.15 # and ori_dist <= 0.1
        return task_success

class PLANNER_AIF_PANDA_REAL(PLANNER_SIMPLE):
    def __init__(self, cfg) -> None:
        self.device = cfg.mppi.device
        self.task = "idle"
        self.curr_goal = torch.zeros(7, device=self.device)
        self.curr_action = "idle"
        # Define the required mdp structures from the templates
        mdp_isCubeAt = isaac_state_action_templates.MDPIsCubeAtReal()

        # Agent with following states [isCubeAt]
        self.ai_agent_task = [ai_agent.AiAgent(mdp_isCubeAt)]
        self.obs = 0
        self.prev_ee_state = torch.zeros(7, device=self.device)
        self.pick_always = False
        self.place_always = False

    def get_obs(self, cube_state, cube_goal, ee_state):
        cube_height_diff = torch.linalg.norm(cube_state[2] - cube_goal[2])
        reach_cost = torch.linalg.norm(ee_state[:3] - cube_state[:3])
        # dist_cost = torch.linalg.norm(cube_goal[:2] - cube_state[:2]) # self.curr_goal
        dist_cost = torch.linalg.norm(self.pre_place_loc[:2] - cube_state[:2])
        ori_ee2cube = skill_utils.get_general_ori_ee2cube(ee_state[3:].view(-1,4), cube_state[3:].view(-1,4), tilt_value=0)
        ori_cost = skill_utils.get_general_ori_cube2goal(cube_goal[3:].view(-1,4), cube_state[3:].view(-1,4))
        print('reach_cost', reach_cost)
        # print("cubbb", cube_state)
        print('dis', dist_cost)
        print('ori', ori_cost)
        # print('ori ee2cube', ori_ee2cube)
        # print("cur goal", self.curr_goal)
        # if cube_height_diff < 0.0001:
        #     self.pick_always = False
        # else:
        #     self.pick_always = True

        if dist_cost + ori_cost < 0.03 or self.place_always:  # 0.035
            self.obs = 2
            self.ai_agent_task[0].set_preferences(np.array([[1], [0], [0], [0]]))
            self.place_always = True
        elif reach_cost < 0.055 or self.pick_always: # 0.055
            self.obs = 1
            self.ai_agent_task[0].set_preferences(np.array([[1], [0], [0], [0]]))
            self.pick_always = True
        elif not self.pick_always:
            self.obs = 0
            self.ai_agent_task[0].set_preferences(np.array([[0], [1], [0], [0]]))

    def update_plan(self, sim):
        sim.step()
        cube_state = sim.get_actor_link_by_name("cubeA", "box")[0, :7]
        cube_goal = sim.get_actor_link_by_name("cubeB", "box")[0, :7]
        left_finger = sim.get_actor_link_by_name("panda", "panda_leftfinger")[0, :7]
        right_finger = sim.get_actor_link_by_name("panda", "panda_rightfinger")[0, :7]
        # print("ee vel", sim.get_actor_link_by_name("panda", "panda_hand")[0, 7:10])
        self.ee_state = (left_finger + right_finger) / 2
        self.pre_place_loc = cube_goal.clone()
        self.pre_place_loc[2] += 0.053
        # print("cube state", cube_state)
        # print("cube_goal", cube_goal)
        self.get_obs(cube_state, cube_goal, self.ee_state)
        # print("obs", self.obs)
        outcome, self.curr_action = adaptive_action_selection.adapt_act_sel(self.ai_agent_task, [self.obs])
        # print('Current action:', self.curr_action)

        self.task = self.curr_action
        if self.curr_action == "reach":
            pass
        if self.curr_action == 'pick':
            self.curr_goal = self.pre_place_loc
        elif self.curr_action == "place":
            pass
    
    def check_task_success(self, sim):
        cube_state = sim.get_actor_link_by_name("cubeA", "box")[0, :7]
        cube_goal = sim.get_actor_link_by_name("cubeB", "box")[0, :7]
        dist_cost = torch.linalg.norm(self.curr_goal[:2] - cube_state[:2])
        flag = False
        if self.task == 'place' and dist_cost < 0.04:
            flag = True
        self.prev_ee_state = self.ee_state.clone()
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