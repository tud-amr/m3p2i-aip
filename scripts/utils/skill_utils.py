import torch

# Calculate the suction force
def calculate_suction(block_pos, robot_pos, num_envs, kp_suction, block_index, bodies_per_env):
    # Calculate the direction and magnitude between the block and robot 
    dir_vector = block_pos - robot_pos # [num_envs, 2]
    magnitude = 1/torch.linalg.norm(dir_vector, dim=1) # [num_envs]
    magnitude = magnitude.reshape([num_envs, 1])

    # Form the suction force
    force = dir_vector*magnitude # [num_envs, 2]
    forces = torch.zeros((num_envs, bodies_per_env, 3), dtype=torch.float32, device='cuda:0', requires_grad=False)
    
    # Start suction only when close
    mask = magnitude[:, :] > 2
    mask = mask.reshape(num_envs)
    # Force on the block
    forces[mask, block_index, 0] = -kp_suction*force[mask, 0]
    forces[mask, block_index, 1] = -kp_suction*force[mask, 1]
    # Opposite force on the robot body
    forces[mask, -1, 0] = kp_suction*force[mask, 0]
    forces[mask, -1, 1] = kp_suction*force[mask, 1]
    # Add clamping to control input
    forces = torch.clamp(forces, min=-500, max=500)

    return forces