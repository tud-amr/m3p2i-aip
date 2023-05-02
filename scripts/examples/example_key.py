from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
import numpy as np
import sys
sys.path.append('../')
from utils import sim_init, skill_utils

class Params:
    def __init__(self) -> None:
        # Make the environment and simulation
        self.allow_viewer = True
        self.num_envs = 1
        self.spacing = 10.0
        self.robot = "panda"          # "point_robot", "boxer", "husky", "albert", "panda", and "heijn"
        self.environment_type = "cube"    # "normal", "battery"
        self.control_type = "vel_control"   # "vel_control", "pos_control", "force_control"


        # Helper variables
        self.suction_active = False       # Activate suction or not when close to purple box
        self.block_index = "None"
        self.kp_suction = 400

        # Time logging
        self.frame_count = 0
        self.next_fps_report = 2.0
        self.t1 = 0
        self.count = 0

        # Print info
        self.print_flag = False

# Get parameters
params = Params()

# Make the environment
gym, sim, viewer, envs, robot_handles = sim_init.make(params.allow_viewer, 
                                                      params.num_envs, 
                                                      params.spacing, 
                                                      params.robot, 
                                                      params.environment_type, 
                                                      params.control_type)

# Acquire states
states_dict = sim_init.acquire_states(gym, sim, params)
dof_states = states_dict["dof_states"]
num_dofs = states_dict["num_dofs"]
dofs_per_robot = states_dict["dofs_per_robot"]
actors_per_env = states_dict["actors_per_env"]
bodies_per_env = states_dict["bodies_per_env"]
block_pos = states_dict["block_pos"]
robot_pos = states_dict["robot_pos"]
cube_state = states_dict["cube_state"]
hand_state = states_dict["hand_state"]
ee_l_state = states_dict["ee_l_state"]
ee_r_state = states_dict["ee_r_state"]

def quaternion_rotation_matrix(Q):
    """
    See https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    # Nvidia uses the quarternion convention of JPL instead of Hamilton
    q0 = Q[:, 3]
    q1 = Q[:, 0]
    q2 = Q[:, 1]
    q3 = Q[:, 2]
    n = Q.size()[0]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = torch.stack((r00, r01, r02, 
                              r10, r11, r12, 
                              r20, r21, r22), dim=1).reshape(n, 3, 3)
                            
    return rot_matrix

# Main loop
while viewer is None or not gym.query_viewer_has_closed(viewer):
    # Step the simulation
    sim_init.step(gym, sim)
    sim_init.refresh_states(gym, sim)

    if params.suction_active:
        # Simulation of a magnetic/suction effect to attach to the box
        suction_force, _, _ = skill_utils.calculate_suction(block_pos, robot_pos, params.num_envs, params.kp_suction, params.block_index, bodies_per_env)
        # print('suc', suction_force)
        # Apply suction/magnetic force
        gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(torch.reshape(suction_force, (params.num_envs*bodies_per_env, 3))), None, gymapi.ENV_SPACE)

    # Respond to keyboard
    sim_init.keyboard_control(gym, sim, viewer, params.robot, num_dofs, params.num_envs, dof_states, params.control_type)

    # print('cube', cube_state)
    # print('ee', ee_l_state)
    ee_mid = (ee_l_state[0,:3] + ee_r_state[0,:3]) / 2
    hand_pos = hand_state[0, :3]
    ee_dir = hand_pos - ee_mid
    ee_dir = ee_dir / torch.norm(ee_dir)
    # print('dir',ee_dir)
    # print('norm', torch.norm(ee_dir))
    cos_theta = ee_dir[2]/torch.norm(ee_dir)
    # print('cos', cos_theta)
    quarternion = ee_l_state[:, 3:7]
    rot_ee = quaternion_rotation_matrix(quarternion)
    # print('qua', quarternion)
    # print('ee', rot_ee)
    zaxis_ee = rot_ee[:, :, 2]
    print('ee z axis', zaxis_ee)
    quarternion_cube = cube_state[:, 3:7]
    rot_cube = quaternion_rotation_matrix(quarternion_cube)
    # print('cube', rot_cube)
    zaxis_cube = rot_cube[:, :, 2]
    print('cube z axis', zaxis_cube)
    cos_theta = torch.sum(torch.mul(zaxis_ee, zaxis_cube), dim=1)
    print('cos', cos_theta)

    # Step rendering
    sim_init.step_rendering(gym, sim, viewer)
    params.next_fps_report, params.frame_count, params.t1 = sim_init.time_logging(gym, sim, params.next_fps_report, params.frame_count, params.t1, params.num_envs)

# Destroy the simulation
sim_init.destroy_sim(gym, sim, viewer)