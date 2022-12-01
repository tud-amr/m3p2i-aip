import numpy as np
from MotionPlanningGoal.goalComposition import GoalComposition
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

def fabrics_point(goal, weight=0.5):
    """
    Initializes the fabric planner for the point robot.
    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.
    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.
    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    """
    goal_dict = {
        "subgoal0": {
            "weight": weight,
            "is_primary_goal": True,
            "indices": [0, 1],
            "parent_link" : 0,
            "child_link" : 1,
            "desired_position": goal,
            "epsilon" : 0.1,
            "type": "staticSubGoal"
        }
    }
    goal_composition = GoalComposition(name="goal", content_dict=goal_dict)

    degrees_of_freedom = 2
    robot_type = "pointRobot"
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    planner = ParameterizedFabricPlanner(
            degrees_of_freedom,
            robot_type,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler
    )
    collision_links = [1]
    self_collision_links = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links,
        self_collision_links,
        goal_composition,
        number_obstacles=1,
    )
    planner.concretize()
    return planner

if __name__ == "__main__":
    from isaacgym import gymapi
    from isaacgym import gymutil
    from isaacgym import gymtorch
    import torch
    from pytorch_mppi import mppi
    from utils import env_conf, sim_init
    import time
    import numpy as np
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

    # Adding Point robot
    num_envs = 1 
    spacing = 10.0
    dt = 0.02

    robot = "point_robot"
    environment_type = "normal"
    control_type = "vel_control"
    gym, sim, viewer, envs, robot_handles = sim_init.make(True, num_envs, spacing, robot, environment_type, control_type, dt=dt)

    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.5, 6, 8), gymapi.Vec3(1.5, 0, 0))
    gym.prepare_sim(sim)

    # Init simulation tensors and torch wrappers (see /docs/programming/tensors.html)
    dof_state =  gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
    actor_root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    # Create fabrics planner
    goal = [3.0, 3.0]
    weight = 1.0
    planner = fabrics_point(goal, weight)

    while True:
        # Compute fabrics action
        dof_state_np = dof_state.cpu().numpy()
        pos = [dof_state_np[0][0], dof_state_np[1][0]]
        vel = [dof_state_np[0][1], dof_state_np[1][1]]
        acc_action = planner.compute_action(
            q=pos,
            qdot=vel,
            x_goal_0=goal,
            weight_goal_0=weight,
            x_obst_0=np.array([2.0, 2.1]),
            weight_obst_0=0.02,
            radius_obst_0=np.array([0.4]),
            radius_body_1=np.array([0.2])
        )
        vel_action = torch.tensor(np.array(vel) + acc_action*dt, dtype=torch.float32, device="cuda:0")

        # Apply action
        gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(vel_action))

        # Step simulator
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)
