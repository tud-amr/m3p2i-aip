import torch, numpy as np, scipy.interpolate as si, time

def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))

def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray

def bspline(c_arr, t_arr=None, n=100, degree=3):
    sample_device = c_arr.device
    sample_dtype = c_arr.dtype
    cv = c_arr.cpu().numpy()

    if(t_arr is None):
        t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
    else:
        t_arr = t_arr.cpu().numpy()
    spl = si.splrep(t_arr, cv, k=degree, s=0.5)
    xx = np.linspace(0, cv.shape[0], n)
    samples = si.splev(xx, spl, ext=3)
    samples = torch.as_tensor(samples, device=sample_device, dtype=sample_dtype)
    return samples

# Track time
def time_tracking(t, cfg):
    actual_dt = time.time() - t
    rt = cfg.isaacgym.dt / actual_dt
    if rt > 1.0:
        time.sleep(cfg.isaacgym.dt - actual_dt)
        actual_dt = time.time() - t
        rt = cfg.isaacgym.dt / actual_dt
    print("FPS: {:.3f}".format(1/actual_dt), "RT: {:.3f}".format(rt))
    return time.time()

# The function to call in sim.py
def check_and_apply_suction(cfg, sim, action):
    suction_force = torch.zeros(10, device=cfg.mppi.device) # the size does not matter
    if check_suction_condition(cfg, sim, action):
        suction_force = calculate_suction(cfg, sim)
        sim.apply_rigid_body_force_tensors(suction_force)
    if suction_force.any() == 0:
        print("no suction...")
    else:
        print("suction!!!")

# Check whether suction is possible
def check_suction_condition(cfg, sim, action):
    if cfg.task not in ['pull', 'push_pull'] or not cfg.suction_active:
        return False
    dir_robot_block = (sim.robot_pos - sim.get_actor_position_by_name("box")[:, :2]).squeeze(0)
    action_align_pull = torch.sum(action * dir_robot_block).item()
    dis_robot_block = torch.linalg.norm(dir_robot_block)

    # Can only pull if the robot is close to block and the action aligns with pulling direction
    # print("dist", dis_robot_block, "align", action_align_pull)
    return dis_robot_block < 0.6 and action_align_pull > 0

# Calculate the suction force
def calculate_suction(cfg, sim):
    # Calculate the direction and magnitude between the block and robot 
    dir_vector = sim.get_actor_position_by_name("box")[:, :2] - sim.robot_pos # [num_envs, 2]
    magnitude = 1/torch.linalg.norm(dir_vector, dim=1) # [num_envs]
    magnitude = magnitude.reshape([sim.num_envs, 1])
    # print('robo', robot_pos)
    # print('mag', magnitude)

    # Form the suction force
    unit_force = dir_vector * magnitude  # [num_envs, 2] Same as the unit direction of pulling force
    forces = torch.zeros((sim.num_envs, sim.bodies_per_env, 3), dtype=torch.float32, device='cuda:0', requires_grad=False)
    
    # Start suction only when close. The different thresholds for real and sim envs are due to the inconsistency 
    # of transferring suction force between sim to real. Among the rollouts, the optimal solution is selected 
    # based on the cost instead of the criteria which one is closest to the block. So the optimal solution 
    # does not mean it is closest to the block. This leads to the inconsistency of suction force.
    if sim.num_envs == 1:
        # For the case of real env, the threshold is lower. 
        # This means the robot and block donot need to be so close to generate the suction
        mask = magnitude[:, :] > 1.5
    else:
        # For the case of simulated rollout env, the threshold is higher.
        # This means the robot and block need to be close enough to generate the suction
        mask = magnitude[:, :] > 1.8
    mask = mask.reshape(sim.num_envs)
    # Force on the block
    block_index = sim._get_actor_index_by_name("box").item()
    forces[mask, block_index, 0] = -cfg.kp_suction * unit_force[mask, 0]
    forces[mask, block_index, 1] = -cfg.kp_suction * unit_force[mask, 1]
    # Opposite force on the robot body
    forces[mask, -1, 0] = cfg.kp_suction * unit_force[mask, 0]
    forces[mask, -1, 1] = cfg.kp_suction * unit_force[mask, 1]
    # Add clamping to control input
    forces = torch.clamp(forces, min=-500, max=500)

    return forces

# Apply forward kinematics
def apply_fk(robot, u):
    '''
    u has the size of [dofs_per_robot]
    '''
    r = 0.08
    L = 2 * 0.157
    if robot == 'boxer':
        # Diff drive fk
        u_fk = u.clone()
        u_fk[0] = (u[0] / r) - ((L * u[1]) / (2 * r))
        u_fk[1] = (u[0] / r) + ((L * u[1]) / (2 * r))
        return u_fk
    elif robot == 'albert':
        u_fk = u.clone()
        u_fk[11] = (u[11] / r) - ((L * u[12]) / (2 * r))
        u_fk[12] = (u[11] / r) + ((L * u[12]) / (2 * r))
        return u_fk
    else:
        return u

# Apply inverse kinematics
def apply_ik(robot, u):
    '''
    u has the size of [num_envs, dofs_per_robot]
    '''
    r = 0.08
    L = 2 * 0.157
    if robot == 'boxer':
        # Diff drive fk
        u_ik = u.clone()
        u_ik[:, 0] = (u[:, 0] / r) - ((L * u[:, 1]) / (2 * r))
        u_ik[:, 1] = (u[:, 0] / r) + ((L * u[:, 1]) / (2 * r))
        return u_ik
    elif robot == 'albert':
        # Diff drive fk
        u_ik = u.clone()
        u_ik[:, 11] = (u[:, 11] / r) - ((L*u[:, 12])/(2*r))
        u_ik[:, 12] = (u[:, 11] / r) + ((L*u[:, 12])/(2*r))
        return u_ik
    else: 
        return u

# Covert a quaternion into a full three-dimensional rotation matrix
def quaternion_rotation_matrix(Q):
    """
    See https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
 
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

# To measure the difference of cube and goal quaternions
def get_ori_cube2goal(cube_quaternion, goal_quatenion):
    """
    Input
    quaternions: tensor in the shape of [num_envs, 4]

    Output 
    return: cost to measure the difference between the two quaternions
    """
    cube_rot_matrix = quaternion_rotation_matrix(cube_quaternion)
    cube_xaxis = cube_rot_matrix[:, :, 0]
    cube_yaxis = cube_rot_matrix[:, :, 1]
    cube_zaxis = cube_rot_matrix[:, :, 2]
    goal_rot_matrix = quaternion_rotation_matrix(goal_quatenion)
    goal_xaxis = goal_rot_matrix[:, :, 0]
    goal_yaxis = goal_rot_matrix[:, :, 1]
    goal_zaxis = goal_rot_matrix[:, :, 2]
    cos_alpha = torch.sum(torch.mul(goal_xaxis, cube_xaxis), dim=1)
    cos_beta = torch.sum(torch.mul(goal_yaxis, cube_yaxis), dim=1)
    cos_gamma = torch.sum(torch.mul(goal_zaxis, cube_zaxis), dim=1)

    return (1-cos_alpha) + (1-cos_beta) + (1-cos_gamma)

# To measure the difference of ee and cube quaternions
def get_ori_ee2cube(ee_quaternion, cube_quaternion):
    ee_rot_matrix = quaternion_rotation_matrix(ee_quaternion)
    ee_xaxis = ee_rot_matrix[:, :, 0]
    ee_yaxis = ee_rot_matrix[:, :, 1]
    ee_zaxis = ee_rot_matrix[:, :, 2]
    cube_rot_matrix = quaternion_rotation_matrix(cube_quaternion)
    cube_xaxis = cube_rot_matrix[:, :, 0]
    cube_yaxis = cube_rot_matrix[:, :, 1]
    cube_zaxis = cube_rot_matrix[:, :, 2]
    cos_alpha = torch.sum(torch.mul(ee_xaxis, cube_xaxis), dim=1)
    cos_theta = torch.sum(torch.mul(ee_zaxis, cube_zaxis), dim=1)
    cos_omega = torch.sum(torch.mul(ee_yaxis, cube_yaxis), dim=1)
    # cos_theta, cos_omega should be close to -1

    return (1+cos_alpha) + (1+cos_theta) + (1+cos_omega)

# A general way to measure the difference of cube and goal quaternions
# so that it fits the case when the cube is flipped and upside down
def get_general_ori_cube2goal(cube_quaternion, goal_quatenion):
    """
    Input
    quaternions: tensor in the shape of [num_envs, 4]

    Output 
    return: cost to measure the difference between the two quaternions
    """
    cube_rot_matrix = quaternion_rotation_matrix(cube_quaternion)
    cube_xaxis = cube_rot_matrix[:, :, 0]
    cube_yaxis = cube_rot_matrix[:, :, 1]
    cube_zaxis = cube_rot_matrix[:, :, 2]
    goal_rot_matrix = quaternion_rotation_matrix(goal_quatenion)
    goal_xaxis = goal_rot_matrix[:, :, 0]
    goal_yaxis = goal_rot_matrix[:, :, 1]
    cos_alpha1 = torch.abs(torch.sum(torch.mul(goal_xaxis, cube_xaxis), dim=1))
    cos_alpha2 = torch.abs(torch.sum(torch.mul(goal_xaxis, cube_yaxis), dim=1))
    cos_alpha3 = torch.abs(torch.sum(torch.mul(goal_xaxis, cube_zaxis), dim=1))
    cost_xaxis = torch.min(torch.stack([1 - cos_alpha1,
                                        1 - cos_alpha2,
                                        1 - cos_alpha3]), dim=0)[0]
    cos_beta1 = torch.abs(torch.sum(torch.mul(goal_yaxis, cube_xaxis), dim=1))
    cos_beta2 = torch.abs(torch.sum(torch.mul(goal_yaxis, cube_yaxis), dim=1))
    cos_beta3 = torch.abs(torch.sum(torch.mul(goal_yaxis, cube_zaxis), dim=1))
    cost_yaxis = torch.min(torch.stack([1 - cos_beta1,
                                        1 - cos_beta2,
                                        1 - cos_beta3]), dim=0)[0]

    return cost_xaxis + cost_yaxis

# To measure the difference of ee and cube quaternions
# so that it fits the case when the cube is flipped and upside down
def get_general_ori_ee2cube(ee_quaternion, cube_quaternion, tilt_value = 0):
    ee_rot_matrix = quaternion_rotation_matrix(ee_quaternion)
    ee_yaxis = ee_rot_matrix[:, :, 1]
    ee_zaxis = ee_rot_matrix[:, :, 2]
    cube_rot_matrix = quaternion_rotation_matrix(cube_quaternion)
    cube_xaxis = cube_rot_matrix[:, :, 0] # red
    cube_yaxis = cube_rot_matrix[:, :, 1] # green
    cube_zaxis = cube_rot_matrix[:, :, 2] # blue

    if tilt_value == 0:
        # Make the ee_zaxis perpendicular to the cube
        cos_theta1 = torch.abs(torch.sum(torch.mul(ee_zaxis, cube_zaxis), dim=1))
        cos_theta2 = torch.abs(torch.sum(torch.mul(ee_zaxis, cube_xaxis), dim=1))
        cos_theta3 = torch.abs(torch.sum(torch.mul(ee_zaxis, cube_yaxis), dim=1))
        cost_zaxis = torch.min(torch.stack([1 - cos_theta1,
                                            1 - cos_theta2,
                                            1 - cos_theta3]), dim=0)[0]
    else:
        # Select the cube axis that is most close to the table xaxis
        stacked_axis = torch.stack([cube_xaxis, cube_yaxis, cube_zaxis])
        indice_list = torch.max(torch.abs(stacked_axis)[:, :, 0], dim=0)[1]
        selected_xaxis = stacked_axis[indice_list[0], :, :]
        # Make the ee_zaxis and selected_xaxis in a tilt anle
        cost_zaxis = torch.abs(tilt_value - torch.sum(torch.mul(ee_zaxis, selected_xaxis), dim=1))

    # Make ee_yaxis align with the cube
    cos_omega1 = torch.abs(torch.sum(torch.mul(ee_yaxis, cube_xaxis), dim=1))
    cos_omega2 = torch.abs(torch.sum(torch.mul(ee_yaxis, cube_yaxis), dim=1))
    cos_omega3 = torch.abs(torch.sum(torch.mul(ee_yaxis, cube_zaxis), dim=1))
    cost_yaxis = torch.min(torch.stack([1 - cos_omega1,
                                        1 - cos_omega2,
                                        1 - cos_omega3]), dim=0)[0]

    # return  5 * cost_zaxis + cost_yaxis #!! for albert
    return  cost_zaxis + cost_yaxis #!! for panda