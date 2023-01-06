import torch

# Calculate the suction force
def calculate_suction(block_pos, robot_pos, num_envs, kp_suction, block_index, bodies_per_env):
    # Calculate the direction and magnitude between the block and robot 
    dir_vector = block_pos - robot_pos # [num_envs, 2]
    magnitude = 1/torch.linalg.norm(dir_vector, dim=1)
    magnitude = torch.reshape(magnitude,[1,num_envs,1])
    direction = dir_vector*magnitude

    force = (magnitude**2)*direction
    forces = torch.zeros((num_envs, bodies_per_env, 3), dtype=torch.float32, device='cuda:0', requires_grad=False)
          
    # Start suction only when close
    mask = magnitude[:, :] > 2  
    mask = torch.reshape(mask, [1,num_envs])
    forces[mask[0],block_index, 0] = -kp_suction*force[0][mask[0],0]
    forces[mask[0],block_index, 1] = -kp_suction*force[0][mask[0],1]
    # Opposite force on the robot body
    forces[mask[0],-1, 0] = kp_suction*force[0][mask[0],0]
    forces[mask[0],-1, 1] = kp_suction*force[0][mask[0],1]

    forces = torch.clamp(forces, min=-500, max=500)

    return forces, -direction[0], mask[0]