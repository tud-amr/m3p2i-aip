import torch
from m3p2i_aip.utils import skill_utils
import  m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper

class Objective(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.multi_modal = cfg.multi_modal
        self.num_samples = cfg.mppi.num_samples
        self.half_samples = int(cfg.mppi.num_samples/2)
        self.device = self.cfg.mppi.device

    def compute_cost(self, sim: wrapper):
        self.goal = self.cfg.goal if torch.is_tensor(self.cfg.goal) else torch.tensor(self.cfg.goal, device=self.device)
        if self.cfg.task == "navigation":
            return self.get_navigation_cost(sim)
        elif self.cfg.task == "push":
            return self.get_push_cost(sim, self.goal)
        elif self.cfg.task == "pull":
            return self.get_pull_cost(sim, self.goal)
        elif self.cfg.task == "push_pull":
            return torch.cat((self.get_push_cost(sim, self.goal)[:self.half_samples], 
                              self.get_pull_cost(sim, self.goal)[self.half_samples:]), dim=0)

    def get_navigation_cost(self, sim: wrapper):
        return torch.linalg.norm(sim.robot_pos - self.goal, axis=1)
    
    def calculate_dist(self, sim: wrapper, block_goal: torch.tensor):
        self.block_pos = sim.get_actor_position_by_name("box")[:, :2]  # x, y position
        robot_to_block = sim.robot_pos - self.block_pos
        block_to_goal = block_goal - self.block_pos

        robot_to_block_dist = torch.linalg.norm(robot_to_block, axis = 1)
        block_to_goal_dist = torch.linalg.norm(block_to_goal, axis = 1)
        
        self.dist_cost = robot_to_block_dist + block_to_goal_dist * 10
        self.cos_theta = torch.sum(robot_to_block*block_to_goal, 1)/(robot_to_block_dist*block_to_goal_dist)

    def get_push_cost(self, sim: wrapper, block_goal: torch.tensor):
        # Calculate dist cost
        self.calculate_dist(sim, block_goal)

        # Force the robot behind block and goal, align_cost is actually cos(theta)+1
        # align_cost = align_weight[robot] * (cos_theta + 1) * 5
        align_cost = torch.zeros(self.num_samples, device=self.device)
        align_cost[self.cos_theta>0] = self.cos_theta[self.cos_theta>0]
        # print('push align', align_cost[:10])
        # ori_cost = skill_utils.get_general_ori_cube2goal(block_quat, goal_quaternion)

        return 3 * self.dist_cost + 1 * align_cost #+ 10 * ori_cost# [num_envs] 31

    def get_pull_cost(self, sim: wrapper, block_goal: torch.tensor):
        self.calculate_dist(sim, block_goal)
        pos_dir = self.block_pos - sim.robot_pos
        robot_to_block_dist = torch.linalg.norm(pos_dir, axis = 1)

        # True means the velocity moves towards block, otherwise means pull direction
        flag_towards_block = torch.sum(sim.robot_vel*pos_dir, 1) > 0

        # simulate a suction to the box
        suction_force = skill_utils.calculate_suction(self.cfg, sim)
        # Set no suction force if robot moves towards the block
        suction_force[flag_towards_block] = 0
        if self.multi_modal:
            suction_force[:self.half_samples] = 0
        sim.apply_rigid_body_force_tensors(suction_force)

        self.calculate_dist(sim, block_goal)

        # Force the robot to be in the middle between block and goal, align_cost is actually 1-cos(theta)
        align_cost = torch.zeros(self.num_samples, device=self.device)
        align_cost[self.cos_theta<0] = -self.cos_theta[self.cos_theta<0]  # (1 - cos_theta)
        # print('pull align', align_cost[-10:])

        # Add the cost when the robot is close to the block and moves towards the block
        vel_cost = torch.zeros(self.num_samples, device=self.device)
        robot_block_close = robot_to_block_dist <= 0.5
        vel_cost[flag_towards_block*robot_block_close] = 0.6

        # ori_cost = skill_utils.get_general_ori_cube2goal(block_quat, goal_quaternion)

        return 3 * self.dist_cost + 3 * vel_cost + 7 * align_cost #+ 10 * ori_cost # [num_envs] 315 

    def get_panda_pick_cost(self, sim, cube_goal_state):
        ee_l_state = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_r_state = sim.get_actor_link_by_name("panda", "panda_rightfinger")
        ee_state = (ee_l_state + ee_r_state) / 2
        cube_state = sim.get_actor_link_by_name("cubeA", "box")

        reach_cost = torch.linalg.norm(ee_state[:,:3] - cube_state[:,:3], axis = 1) 
        goal_cost = torch.linalg.norm(cube_goal_state[:3] - cube_state[:,:3], axis = 1) #+ 2*torch.abs(block_goal[2] - block_state[:,2])
        # Close the gripper when close to the cube
        gripper_dist = torch.linalg.norm(ee_l_state[:, :3] - ee_r_state[:, :3], axis=1)
        gripper_cost = 2 * (1 - gripper_dist)
        threshold_gripper = {'panda':0.1, 'albert':0.08}
        gripper_cost[reach_cost < 0.1] = 0

        manip_cost = torch.zeros_like(reach_cost)
        
        # Compute the orientation cost
        cube_quaternion = cube_state[:, 3:7]
        goal_quatenion = cube_goal_state[3:7].repeat(self.num_samples).view(self.num_samples, 4)
        # To make the cube fit the goal's orientation well
        ori_cube2goal = skill_utils.get_general_ori_cube2goal(cube_quaternion, goal_quatenion) 
        ori_cost = 3 * ori_cube2goal

        # Compute the tilt value between ee and cube
        tilt_cost = self.get_pick_tilt_cost(sim)
        tilt_cost[reach_cost<=0.05] = 0
        weight_goal = {True:15, False:5}
        total_cost = 0.2 * manip_cost + 10 * reach_cost + weight_goal[self.multi_modal] * goal_cost + ori_cost + gripper_cost + tilt_cost

        return  total_cost #+ align_cost multiply 10*reach_cost when using mppi_mode == storm

    def get_pick_tilt_cost(self, sim):
        # This measures the cost of the tilt angle between the end effector and the cube
        ee_l_state = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_quaternion = ee_l_state[:, 3:7]
        cubeA_pos = sim.get_actor_position_by_name("cubeA")
        cubeA_ori = sim.get_actor_orientation_by_name("cubeA")
        # cube_quaternion = cube_state[:, 3:7]
        if not self.multi_modal:
            # To make the z-axis direction of end effector to be perpendicular to the cube surface
            ori_ee2cube = skill_utils.get_general_ori_ee2cube(ee_quaternion, cubeA_ori, tilt_value=0)
        else:
            # To combine costs of different tilt angles
            cost_1 = skill_utils.get_general_ori_ee2cube(ee_quaternion[:self.half_samples], 
                                                            cubeA_ori[:self.half_samples], tilt_value = 0)
            cost_2 = skill_utils.get_general_ori_ee2cube(ee_quaternion[self.half_samples:], 
                                                            cubeA_ori[self.half_samples:], tilt_value = 0.5) #0.9
            ori_ee2cube =  torch.cat((cost_1, cost_2), dim=0)

        return 3 * ori_ee2cube

    def get_panda_place_cost(self, sim, ee_goal):
        ee_l_state = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_r_state = sim.get_actor_link_by_name("panda", "panda_rightfinger")

        gripper_dist = torch.linalg.norm(ee_l_state[:, :3] - ee_r_state[:, :3], axis=1)
        gripper_cost = 1 - gripper_dist
        ee_state = (ee_l_state + ee_r_state) / 2
        reach_cost = torch.linalg.norm(ee_state[:,:7] - ee_goal[:7], axis=1)

        # If gripper is not fully open, no reach cost
        reach_cost[gripper_dist <= 0.078] = 0
        # If gripper is fully open, no gripper cost, retract the arm
        gripper_cost[gripper_dist > 0.078] = 0
        return 10 * gripper_cost + 10 * reach_cost

    def get_motion_cost(self, sim, t):
        if self.cfg.env_type == 'point_env':   
            obs_force = sim.get_actor_contact_forces_by_name("dyn-obs", "box") # [num_envs, 3]
        elif self.cfg.env_type == 'panda_env':
            obs_force = sim.get_actor_contact_forces_by_name("table", "box")
        coll_cost = torch.sum(torch.abs(obs_force[:, :2]))
        # Binary check for collisions.
        coll_cost[coll_cost>0.1] = 1
        coll_cost[coll_cost<=0.1] = 0

        # Avoid dynamic obstacle
        penalty_factor = 2 # the larger the factor, the more penalty to geting close to the obs
        allow_dyn_obs = False
        dyn_obs_cost = self._predict_dyn_obs(penalty_factor, t+1, sim) if allow_dyn_obs else 0

        return 1000*coll_cost + dyn_obs_cost

    def _predict_dyn_obs(self, factor, t, sim):
        dyn_obs_pos = sim.get_actor_position_by_name("dyn-obs")
        # Obs boundary [-2.5, 1.5] <--> [-1.5, 2.5]
        obs_lb = torch.tensor([-2.5, 1.5], device=self.device)
        obs_ub = torch.tensor([-1.5, 2.5], device=self.device)
        dyn_obs_vel = torch.clamp(dyn_obs_vel, min = -0.001, max = 0.001)
        pred_pos = dyn_obs_pos + t * dyn_obs_vel * 10
        # Check the prec_pos and boundary
        exceed_ub = pred_pos[:, 1] > obs_ub[1]
        exceed_lb = pred_pos[:, 1] < obs_lb[1]
        pred_pos[exceed_ub] = 2 * obs_ub - pred_pos[exceed_ub]
        pred_pos[exceed_lb] = 2 * obs_lb - pred_pos[exceed_lb]
        # Compute the cost
        dyn_obs_cost = factor * torch.exp(-torch.norm(pred_pos - sim.robot_pos, dim=1))

        return dyn_obs_cost