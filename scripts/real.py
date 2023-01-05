from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from fusion_mppi import mppi, fusion_mppi
from utils import env_conf, sim_init, data_transfer
import time
import copy
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)


class IsaacgymMppiRos:
    def __init__(self):
        self.init_isaacgym_mppi()

        self.frequency = 10
        self.rate = rospy.Rate(self.frequency)


        rospy.init_node(name="mppi_point")
        rospy.Subscriber('/natnet_ros/Heijn/pose', Odometry, self.state_cb, queue_size=1)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    def state_cb(self, msg):
        self.state = torch.tensor([
            msg.pose.pose.position.x,
            msg.twist.twist.linear.x,
            msg.pose.pose.position.y,
            msg.twist.twist.linear.y,
        ])

    def init_isaacgym_mppi(self):
        # Make the environment and simulation
        allow_viewer = False
        num_envs = 100
        spacing = 10.0
        robot = "point_robot"               # choose from "point_robot", "boxer", "albert"
        environment_type = "normal"         # choose from "normal", "battery"
        control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"
        self.gym, self.sim, self.viewer, envs, robot_handles = sim_init.make(allow_viewer, num_envs, spacing, robot, environment_type, control_type, dt=1/self.frequency)

        # Acquire states
        dof_states, num_dofs, num_actors, root_states = sim_init.acquire_states(self.gym, self.sim, print_flag=False)

        self.root_states = copy.deepcopy(root_states)

        # Creater mppi object
        self.mppi = fusion_mppi.FUSION_MPPI(
            dynamics=None, 
            running_cost=None, 
            nx=4, 
            noise_sigma = torch.tensor([[5, 0], [0, 5]], device="cuda:0", dtype=torch.float32),
            num_samples=num_envs, 
            horizon=15,
            lambda_=0.1, 
            device="cuda:0", 
            u_max=torch.tensor([0.5, 0.5]),
            u_min=torch.tensor([-0.5, -0.5]),
            step_dependent_dynamics=True,
            terminal_state_cost=None,
            sample_null_action=False,
            use_priors=False,
            robot_type=robot,
            u_per_command=15
        )

    def run(self):
        with not rospy.is_shutdown():
            # Reset the simulator to requested state
            s = self.state # [x, v_x, y, v_y]
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(s))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            sim_init.refresh_states(self.gym, self.sim)

            # Update gym in mppi
            self.mppi.update_gym(self.gym, self.sim, self.viewer)

            # Compute optimal action and send to real simulator
            action = self.mppi.command(s)[0]

            # Publish action command
            command = Twist()
            command.linear.x = action[0]
            command.linear.y = action[1]

            self.pub.publish()

            self.rate.sleep()


