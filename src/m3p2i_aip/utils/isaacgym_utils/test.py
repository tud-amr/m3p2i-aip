import  m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper

sim = wrapper.IsaacGymWrapper(
        wrapper.IsaacGymConfig,
        env_type="point_env",
        num_envs=10,
        viewer=True,
    )

for i in range(2000):
    sim.step()
    sim.keyboard_control()
    if True: 
        print("dof states", sim._dof_state)
        print("root states", sim._root_state)
        # print("rb states", sim._rigid_body_state)
        if sim.env_type == "panda_env":
            cubeA_pos = sim.get_actor_position_by_name("cubeA")
            cubeA_ori = sim.get_actor_orientation_by_name("cubeA")
            cubeB_pos = sim.get_actor_position_by_name("cubeB")
            cubeB_ori = sim.get_actor_orientation_by_name("cubeB")
            left_finger = sim.get_actor_link_by_name("panda", "panda_leftfinger")
            right_finger = sim.get_actor_link_by_name("panda", "panda_rightfinger")
            print("cubeA_pos", cubeA_pos)
            print("cubeA_ori", cubeA_ori)
            print("cubeB_pos", cubeB_pos)
            print("cubeB_ori", cubeB_ori)
            print("left_finger:", left_finger)
            print("right_finger:", right_finger)
        elif sim.env_type == "point_env":
            box_pos = sim.get_actor_position_by_name("box")
            box_ori = sim.get_actor_orientation_by_name("box")
            dyn_obs_pos = sim.get_actor_position_by_name("dyn-obs")
            goal_pos = sim.get_actor_position_by_name("goal")
            print("point_pos", sim.robot_pos)
            print("point_vel", sim.robot_vel)
            print("box_pos", box_pos)
            print("dyn_obs_pos", dyn_obs_pos)
            # print("goal_pos", goal_pos)