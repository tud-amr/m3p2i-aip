from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from utils import sim_init

# Make the environment and simulation
allow_viewer = True
num_envs = 4
spacing = 10.0
robot = "point_robot"                     # "point_robot", "boxer", "husky", and "albert"
environment_type = "normal"            # choose from "normal", "battery"
control_type = "vel_control"        # choose from "vel_control", "pos_control", "force_control"

# Helper variables
suction_active = True       # Activate suction or not when close to purple box
bodies_per_env = 18
box_index = 7
kp = 400

# Time logging
frame_count = 0
next_fps_report = 2.0
t1 = 0
count = 0

gym, sim, viewer, envs, robot_handles = sim_init.make(allow_viewer, num_envs, spacing, robot, environment_type, control_type)

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 6, 8), gymapi.Vec3(1.5, 0, 0))
gym.prepare_sim(sim)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = torch.reshape(gymtorch.wrap_tensor(_dof_states), (num_envs, 4))
num_dofs = gym.get_sim_dof_count(sim)

def calculate_suction(v1, v2):
    dir_vector = v1 - v2
    magnitude = 1/torch.linalg.norm(dir_vector, dim=1)
    magnitude = torch.reshape(magnitude,[1,num_envs,1])
    direction = dir_vector/magnitude

    force = (magnitude**2)*direction
    forces = torch.zeros((num_envs, bodies_per_env, 3), dtype=torch.float32, device='cuda:0', requires_grad=False)

    if suction_active:   # Start suction only when close
        for i in range(num_envs):
            if magnitude[0,i] > 2:
                forces[i,box_index, 0]= -kp*force[0][i,0]
                forces[i,box_index, 1] = -kp*force[0][i,1]
                # Opposite force on the robot body
                forces[i,-1, 0] = kp*force[0][i,0]
                forces[i,-1, 1] = kp*force[0][i,1]

    forces = torch.clamp(forces, min=-500, max=500)

    return forces

# Main loop
while viewer is None or not gym.query_viewer_has_closed(viewer):
    # Step the simulation
    sim_init.step(gym, sim)
    sim_init.refresh_states(gym, sim)

    actor_root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    root_positions = torch.reshape(actor_root_state[:, 0:2], (num_envs, bodies_per_env-2, 2))
    dof_pos = dof_states[:,[0,2]]
    print('dof', dof_pos)
    print('root', root_positions[:, box_index, :])
    # simulation of a magnetic/suction effect to attach to the box
    suction_force = calculate_suction(root_positions[:, box_index, :], dof_pos)
    
    # Apply suction/magnetic force
    gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(torch.reshape(suction_force, (num_envs*bodies_per_env, 3))), None, gymapi.ENV_SPACE)

    # Respond to keyboard
    sim_init.keyboard_control(gym, sim, viewer, robot, num_dofs, num_envs, dof_states, control_type)

    # Step rendering
    sim_init.step_rendering(gym, sim, viewer)
    next_fps_report, frame_count, t1 = sim_init.time_logging(gym, sim, next_fps_report, frame_count, t1, num_envs)

# Destroy the simulation
sim_init.destroy_sim(gym, sim, viewer)
