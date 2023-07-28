from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from fusion_mppi import mppi, fusion_mppi
from utils import env_conf, sim_init, data_transfer
import time
import copy
import rospy
import tf
from tf.transformations import euler_from_quaternion
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist, Quaternion
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)
from params import params_panda as params
from active_inference import task_planner
from utils import skill_utils
import actionlib
from franka_gripper.msg import MoveAction, MoveActionGoal, MoveGoal, GraspAction, GraspActionGoal, GraspGoal

# Helpers
def _it(self):
    yield self.x
    yield self.y
    yield self.z
    yield self.w
# Quaternion.__iter__ = _it

class IsaacgymMppiRos:
    def __init__(self, params):
        rospy.init_node(name="mppi_panda")
        self.params = params
        self.frequency = 20
        self.rate = rospy.Rate(self.frequency)
        self.init_isaacgym_mppi()
        self.tf_listener = tf.TransformListener()

        rospy.Subscriber('/joint_states_filtered', JointState, self.state_cb_arm, queue_size=1)
        rospy.Subscriber('/joint_states', JointState, self.state_cb_fingers, queue_size=1)
        # rospy.Subscriber('/optitrack_state_estimator/BlueBlock/state', Odometry, self.cubeA_cb, queue_size=1)
        # rospy.Subscriber('/optitrack_state_estimator/RedBlock/state', Odometry, self.cubeB_cb, queue_size=1)

        self.pub = rospy.Publisher('/panda_joint_velocity_controller/command', 
                                   Float64MultiArray, queue_size=10)
        time.sleep(1)
        
    # Get dof states of the panda arm
    def state_cb_arm(self, msg):
        self.q_arm = torch.tensor(msg.position[5:], device='cuda:0')
        self.qdot_arm = torch.tensor(msg.velocity[5:], device='cuda:0')

    # Get dof states of the panda arm
    def state_cb_fingers(self, msg):
        self.q_fingers = torch.tensor(msg.position[-2:], device='cuda:0')
        self.qdot_fingers = torch.tensor(msg.velocity[-2:], device='cuda:0')                             

    def get_state_cb(self, cube_frame):
        state = torch.zeros(7, device='cuda:0')
        try: 
            (pos, quat) = self.tf_listener.lookupTransform(
                "/Panda", cube_frame, rospy.Time(0)
            )
            state = torch.tensor(pos+quat, device='cuda:0')
            return state
        except(
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException
        ):
            print('Something wrong')

    def init_isaacgym_mppi(self):
        # Make the environment and simulation
        self.gym, self.sim, self.viewer, envs, _ = sim_init.make(params.allow_viewer, params.num_envs, 
                                                                 params.spacing, params.robot, 
                                                                 params.environment_type, dt=1/self.frequency)

        # Acquire states
        states_dict = sim_init.acquire_states(self.gym, self.sim, params)
        self.dofs_per_robot = states_dict["dofs_per_robot"]
        self.actors_per_env = states_dict["actors_per_env"]
        self.bodies_per_env = states_dict["bodies_per_env"]
        self.cube_state = states_dict["cube_state"] # cubeA
        self.cube_goal_state = states_dict["cube_goal_state"] # cubeB
        self.ee_l_state = states_dict["ee_l_state"]
        self.ee_r_state = states_dict["ee_r_state"]
        self.ee_state = states_dict["ee_state"]
        self.root_states = states_dict["root_states"]

        # Choose the task planner
        # self.task_planner = task_planner.PLANNER_AIF_PANDA_REAL()
        self.task_planner = task_planner.PLANNER_AIF_PANDA_REACTIVE()

        # Choose the motion planner
        self.motion_planner = fusion_mppi.FUSION_MPPI(
                            params = params,
                            dynamics = None, 
                            running_cost = None, 
                            nx = params.nx, 
                            noise_sigma = params.noise_sigma,
                            num_samples = params.num_envs, 
                            horizon = params.horizon,
                            lambda_ = params.lambda_, 
                            device = params.device, 
                            u_max = params.u_max,
                            u_min = params.u_min,
                            step_dependent_dynamics = params.step_dependent_dynamics,
                            terminal_state_cost = params.terminal_state_cost,
                            sample_null_action = params.sample_null_action,
                            use_priors = params.use_priors,
                            use_vacuum = params.suction_active,
                            robot_type = params.robot,
                            u_per_command = params.u_per_command,
                            actors_per_env = self.actors_per_env,
                            env_type = params.environment_type,
                            bodies_per_env = self.bodies_per_env,
                            filter_u = params.filter_u
                            )
        self.motion_planner.set_mode(
            mppi_mode = 'halton-spline', # 'halton-spline', 'simple'
            sample_method = 'halton',    # 'halton', 'random'
            multi_modal = False           # True, False
        )
        self.prefer_pull = -1
        self.cubeA_index = 2
        self.cubeB_index = 3
        self.magicwand_index = 4
        self.flag = True
        self.grasp_flag = True
        self.open_flag = True
        
        self.close_client = actionlib.SimpleActionClient('franka_gripper/move', MoveAction)
        self.open_client = actionlib.SimpleActionClient('franka_gripper/grasp', GraspAction)

        self.close_client.wait_for_server()
        self.open_client.wait_for_server()


    def tamp_interface(self):
        # Update task and goal in the task planner
        gripper_dist = self.q_fingers[0] + self.q_fingers[1]
        self.task_planner.update_plan(self.cubeA_state[:7], 
                                      self.cubeB_state[:7], 
                                      self.ee_state_real[:7],
                                      gripper_dist)

        # Update task and goal in the motion planner
        # print('task:', self.task_planner.task, 'goal:', self.task_planner.curr_goal)
        self.motion_planner.update_task(self.task_planner.task, self.task_planner.curr_goal)

        # Update params in the motion planner
        self.params = self.motion_planner.update_params(params, self.prefer_pull)

        # Check task succeeds or not
        task_success = self.task_planner.check_task_success(self.ee_state_real[:7])
        return task_success

    def run(self):
        while not rospy.is_shutdown():
            # Reset states of the arm
            dof_state = torch.zeros((9, 2), device='cuda:0')
            dof_state[:7, 0] = self.q_arm
            dof_state[7:, 0] = self.q_fingers
            # print('finger', self.q_fingers)
            dof_state[:7, 1] = self.qdot_arm
            dof_state[7:, 1] = self.qdot_fingers
            # print('finger_vel', self.qdot_fingers)
            # print('dof', dof_state)
            s = dof_state.repeat(params.num_envs, 1) # [j1, v_j1, yj2, v_j2, j3, v_j3, ..., ]
            if self.flag:
                # print('kkkk')
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(s))
                # self.flag = False
            # Reset the states of the cubes
            self.cubeA_state = self.get_state_cb("/BlueBlock")
            self.cubeB_state = self.get_state_cb("/RedBlock")
            self.magicwand_state = self.get_state_cb("/MagicWand")
            self.root_states = self.root_states.reshape([params.num_envs, self.actors_per_env, 13])
            self.root_states[:, self.cubeA_index, :7] = self.cubeA_state
            self.root_states[:, self.cubeA_index, 7:] = 0
            self.root_states[:, self.cubeB_index, :7] = self.cubeB_state
            self.root_states[:, self.cubeB_index, 7:] = 0 
            self.root_states[:, self.magicwand_index, :7] = self.magicwand_state
            self.root_states[:, self.magicwand_index, 7:] = 0
            self.root_states = self.root_states.reshape([params.num_envs * self.actors_per_env, 13])
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.refresh_actor_root_state_tensor(self.sim)
            # Get state of ee
            self.ee_state_real = self.get_state_cb("/panda_ee_low")
            # print('A', self.cubeA_state)
            # print('sim A', self.cube_state[0,:])
            # print('B', self.cubeB_state)
            # print('ee real', self.ee_state_real)
            # print('ee sim', self.ee_state[:3, :7])
            # print('sim', torch.linalg.norm(self.ee_state[0, :3] - self.cube_state[0, :3]))
            # print('real', torch.linalg.norm(self.ee_state_real[:3] - self.cubeA_state[:3]))
            # print('magic', self.magicwand_state)

            cubeA_quat = self.cubeA_state[3:7].view(1, 4)
            cubeB_quat = self.cubeB_state[3:7].view(1, 4)
            ee_sim_quat = self.ee_state[0, 3:7].view(1, 4)
            ee_real_quat = self.ee_state_real[3:7].view(1, 4)
            ee_l_state = self.ee_l_state[0, 3:7].view(1, 4)
            ori_cost = skill_utils.get_general_ori_cube2goal(cubeA_quat, cubeB_quat)
            ori_cost2 = skill_utils.get_general_ori_cube2goal(cubeA_quat, ee_sim_quat)
            ori_cost3 = skill_utils.get_general_ori_cube2goal(cubeA_quat, ee_real_quat)
            ori_cost4 = skill_utils.get_general_ori_cube2goal(ee_l_state, ee_real_quat)
            # print('ori', ori_cost)
            # print('ori sim to cube', ori_cost2)
            # print('ori real to cube', ori_cost3)
            # print('l to real', ori_cost4)

            # Step the simulation
            sim_init.step(self.gym, self.sim) # !!
            sim_init.refresh_states(self.gym, self.sim)

            # Update TAMP interface
            task_success = self.tamp_interface()

            # Update gym in mppi
            self.motion_planner.update_gym(self.gym, self.sim, self.viewer)

            # Compute optimal action and send to real simulator
            action = np.array(self.motion_planner.command(s)[0].cpu())
            # print('hyy', action[:])
            # Griiper command should be discrete? TODO

            # Publish action command 
            command = Float64MultiArray()
            if task_success:
                command.data = [0] * 7
            else:
                command.data = action[:7]
            if not params.allow_viewer:
                self.pub.publish(command)

                # Send the command of gripper
                # >0.18 for the no collision cost
                # self.task_planner.task == 'pick'
                if self.task_planner.task in ['pick'] and self.grasp_flag: #!!!
                    print('ccccccclose')
                    grasp_goal = MoveGoal()
                    grasp_goal.width = 0.0
                    grasp_goal.speed = 0.1
                    # print(grasp_goal)
                    self.close_client.send_goal(grasp_goal)
                    self.grasp_flag = False # Set to false when pick once
                    self.open_flag = True
                elif self.task_planner.task in ['', 'place', 'idle_success'] and self.open_flag:
                    print('oooooooooopen')
                    open_goal = GraspGoal()
                    open_goal.width = 0.38
                    open_goal.speed = 0.1
                    open_goal.force = 0.1
                    self.open_client.send_goal(open_goal)
                    self.grasp_flag = True
                    self.open_flag = False

            self.rate.sleep()


test = IsaacgymMppiRos(params)
test.run()

