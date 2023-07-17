from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from fusion_mppi import mppi, fusion_mppi
from utils import env_conf, sim_init, data_transfer
import time
import copy
import rospy
from tf.transformations import euler_from_quaternion
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist, Quaternion
from nav_msgs.msg import Odometry
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)
from params import params_panda as params
from active_inference import task_planner

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
        self.frequency = 10
        self.rate = rospy.Rate(self.frequency)
        self.init_isaacgym_mppi()

        rospy.Subscriber('/optitrack_state_estimator/Panda/state', Odometry, self.state_cb, queue_size=1)
        rospy.Subscriber('/optitrack_state_estimator/CubeA/state', Odometry, self.cubeA_cb, queue_size=1)
        rospy.Subscriber('/optitrack_state_estimator/CubeB/state', Odometry, self.cubeB_cb, queue_size=1)

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        time.sleep(1)
        
    # TODO get dof states of the panda arm
    def state_cb(self, msg):
        _, _, yaw = euler_from_quaternion(list(msg.pose.pose.orientation))
        self.state = torch.tensor([
            msg.pose.pose.position.x,
            msg.twist.twist.linear.x,
            msg.pose.pose.position.y,
            msg.twist.twist.linear.y,
            yaw,
            msg.twist.twist.angular.z,
            ], device='cuda:0')
        self.R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        # get states of the end effector TODO
        # self.ee_state


    def cubeA_cb(self, msg):
        _, _, yaw = euler_from_quaternion(list(msg.pose.pose.orientation))
        self.cubeA_state = torch.tensor([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            0,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
            ], device='cuda:0')
        self.cubeA_R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    
    def cubeB_cb(self, msg):
        _, _, yaw = euler_from_quaternion(list(msg.pose.pose.orientation))
        self.cubeB_state = torch.tensor([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            0,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
            ], device='cuda:0')
        self.cubeB_R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

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
        self.cube_goal_state_new = self.cube_goal_state[0, :7].clone()
        self.cube_goal_state_new[2] += 0.06

        # Choose the task planner
        # self.task_planner = task_planner.PLANNER_PICK("pick", self.cube_goal_state_new)
        self.task_planner = task_planner.PLANNER_AIF_PANDA()

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
        self.cubeA_index = 3
        self.cubeB_index = 4
    
    def tamp_interface(self):
        # Update task and goal in the task planner
        self.task_planner.update_plan(self.cubeA_state[:7], 
                                      self.cubeB_state[:7], 
                                      self.ee_state[:7])

        # Update task and goal in the motion planner
        # print('task:', self.task_planner.task, 'goal:', self.task_planner.curr_goal)
        self.motion_planner.update_task(self.task_planner.task, self.task_planner.curr_goal)

        # Update params in the motion planner
        self.params = self.motion_planner.update_params(params, self.prefer_pull)

        # Check task succeeds or not
        task_success = self.task_planner.check_task_success(self.ee_state[:7])
        return task_success

    def run(self):
        while not rospy.is_shutdown():
            # Reset the simulator to requested state
            # Reset states of the arm TODO
            s = self.state.repeat(params.num_envs, 1) # [x, v_x, y, v_y, yaw, v_yaw]
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(s))
            # Reset the states of the cubes
            self.root_states = self.root_states.reshape([params.num_envs, self.actors_per_env, 13])
            self.root_states[:, self.cubeA_index, :] = self.cubeA_state
            self.root_states[:, self.cubeB_index, :] = self.cubeB_state
            self.root_states = self.root_states.reshape([params.num_envs * self.actors_per_env, 13])
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            sim_init.step(self.gym, self.sim) # !!
            sim_init.refresh_states(self.gym, self.sim)

            # Update TAMP interface
            task_success = self.tamp_interface()

            # Update gym in mppi
            self.motion_planner.update_gym(self.gym, self.sim, self.viewer)

            # Compute optimal action and send to real simulator
            action = np.array(self.motion_planner.command(s)[0].cpu())
            # Griiper command should be discrete? TODO

            # TODO
            # # Tranform world->robot frame
            # action[:2] = self.R.T.dot(action[:2].T)
            # # Publish action command 
            command = Twist()
            command.linear.x = action[0]
            command.linear.y = action[1]
            command.angular.z = action[2]

            self.pub.publish(command)

            self.rate.sleep()


# test = IsaacgymMppiRos(params)
# test.run()

