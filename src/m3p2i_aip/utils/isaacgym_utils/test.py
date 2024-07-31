import  m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper

sim = wrapper.IsaacGymWrapper(
        wrapper.IsaacGymConfig,
        env_type="point_env",
        num_envs=1,
        viewer=True,
    )

for i in range(2000):
    sim.step()