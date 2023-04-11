import torch

# Calculate the suction force
def calculate_suction(block_pos, robot_pos, num_envs, kp_suction, block_index, bodies_per_env):
    # Calculate the direction and magnitude between the block and robot 
    dir_vector = block_pos - robot_pos # [num_envs, 2]
    magnitude = 1/torch.linalg.norm(dir_vector, dim=1) # [num_envs]
    magnitude = magnitude.reshape([num_envs, 1])
    # print('robo', robot_pos)
    # print('mag', magnitude)

    # Form the suction force
    unit_force = dir_vector*magnitude  # [num_envs, 2] Same as the unit direction of pulling force
    forces = torch.zeros((num_envs, bodies_per_env, 3), dtype=torch.float32, device='cuda:0', requires_grad=False)
    
    # Start suction only when close
    # The different thresholds for real and sim envs are due to the inconsistency of 
    # transferring suction force between sim to real. Among the rollouts, the optimal
    # solution is selected based on the cost instead of the criteria which one is closest to the block. 
    # So the optimal solution does not mean it is closest to the block. This leads to the inconsistency of suction force.
    if num_envs == 1:
        # For the case of real env, the threshold is lower. 
        # This means the robot and block donot need to be so close to generate the suction
        mask = magnitude[:, :] > 1.5
    else:
        # For the case of simulated rollout env, the threshold is higher.
        # This means the robot and block need to be close enough to generate the suction
        mask = magnitude[:, :] > 1.8
    mask = mask.reshape(num_envs)
    # Force on the block
    forces[mask, block_index, 0] = -kp_suction*unit_force[mask, 0]
    forces[mask, block_index, 1] = -kp_suction*unit_force[mask, 1]
    # Opposite force on the robot body
    forces[mask, -1, 0] = kp_suction*unit_force[mask, 0]
    forces[mask, -1, 1] = kp_suction*unit_force[mask, 1]
    # Add clamping to control input
    forces = torch.clamp(forces, min=-500, max=500)

    return forces, -unit_force, mask

# Apply forward kinematics
def apply_fk(robot, u):
    '''
    u has the size of [dofs_per_robot]
    '''
    if robot == 'boxer':
        r = 0.08
        L = 2 * 0.157
        # Diff drive fk
        u_fk = u.clone()
        u_fk[0] = (u[0] / r) - ((L * u[1]) / (2 * r))
        u_fk[1] = (u[0] / r) + ((L * u[1]) / (2 * r))
        return u_fk
    else:
        return u

# Apply inverse kinematics
def apply_ik(robot, u):
    '''
    u has the size of [num_envs, dofs_per_robot]
    '''
    if robot == 'boxer':
        r = 0.08
        L = 2 * 0.157
        # Diff drive fk
        u_ik = u.clone()
        u_ik[:, 0] = (u[:, 0] / r) - ((L * u[:, 1]) / (2 * r))
        u_ik[:, 1] = (u[:, 0] / r) + ((L * u[:, 1]) / (2 * r))
        return u_ik
    else: 
        return u

 
