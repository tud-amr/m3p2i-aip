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

# Helpers
def _it(self):
    yield self.x
    yield self.y
    yield self.z
    yield self.w
Quaternion.__iter__ = _it


class IsaacgymMppiRos:
    def __init__(self):
        rospy.init_node(name="mppi_point")
        self.frequency = 10
        self.rate = rospy.Rate(self.frequency)
        self.init_isaacgym_mppi()


        rospy.Subscriber('/optitrack_state_estimator/Heijn/state', Odometry, self.state_cb, queue_size=1)
        rospy.Subscriber('/optitrack_state_estimator/Crate/state', Odometry, self.block_cb, queue_size=1)

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        time.sleep(1)
        
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

    def block_cb(self, msg):
        _, _, yaw = euler_from_quaternion(list(msg.pose.pose.orientation))
        self.block_state = torch.tensor([
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
        self.block_R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    def init_isaacgym_mppi(self):
        # Make the environment and simulation
        allow_viewer = False
        self.num_envs = 200
        spacing = 10.0
        robot = "heijn"               # choose from "point_robot", "boxer", "albert"
        environment_type = "lab"         # choose from "normal", "battery"
        control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
        self.gym, self.sim, self.viewer, envs, robot_handles = sim_init.make(allow_viewer, self.num_envs, spacing, robot, environment_type, control_type, dt=1/self.frequency)

        # Acquire states
        dof_states, num_dofs, self.num_actors, root_states = sim_init.acquire_states(self.gym, self.sim, print_flag=False)

        self.root_states = copy.deepcopy(root_states)

        # Creater mppi object
        self.mppi = fusion_mppi.FUSION_MPPI(
            dynamics=None, 
            running_cost=None, 
            nx=6, 
            noise_sigma = torch.tensor([[15, 0, 0], [0, 15, 0], [0, 0, 15]], device="cuda:0", dtype=torch.float32),
            num_samples=self.num_envs, 
            horizon=20,
            lambda_=0.3, 
            device="cuda:0", 
            u_max=torch.tensor([0.6, 0.6, 1.0]),
            u_min=torch.tensor([-0.6, -0.6, -1.0]),
            step_dependent_dynamics=True,
            terminal_state_cost=None,
            sample_null_action=True,
            use_priors=False,
            use_vacuum = False,
            robot_type=robot,
            u_per_command=20,
            actors_per_env=int(self.num_actors/self.num_envs),
            env_type=environment_type,
            bodies_per_env=self.gym.get_env_rigid_body_count(envs[0])
        )

    def run(self):
        while not rospy.is_shutdown():
            # Reset the simulator to requested state
            s = self.state.repeat(self.num_envs, 1) # [x, v_x, y, v_y, yaw, v_yaw]
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(s))
            block_index = 4
            for i in range(self.num_envs):
                self.root_states[i*int(self.num_actors/self.num_envs) + block_index] = self.block_state
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            sim_init.refresh_states(self.gym, self.sim)

            # Update gym in mppi
            self.mppi.update_gym(self.gym, self.sim, self.viewer)

            # Compute optimal action and send to real simulator
            action = np.array(self.mppi.command(s)[0].cpu())

            # Tranform world->robot frame
            action[:2] = self.R.T.dot(action[:2].T)
            # Publish action command 
            command = Twist()
            command.linear.x = action[0]
            command.linear.y = action[1]
            command.angular.z = action[2]

            self.pub.publish(command)

            self.rate.sleep()


test = IsaacgymMppiRos()
test.run()

