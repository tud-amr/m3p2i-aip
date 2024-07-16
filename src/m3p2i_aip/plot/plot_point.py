from m3p2i_aip.utils import path_utils, skill_utils
import numpy as np
import torch
import plotly.graph_objects as go

file_path_1 = path_utils.get_plot_path() +'/point/case2_halton_push_coll.npy'
file_path_2 = path_utils.get_plot_path() +'/point/case2_halton_pull_coll.npy'
file_path_3 = path_utils.get_plot_path() +'/point/corner1_push.npy'
file_path_4 = path_utils.get_plot_path() +'/point/corner1_pull.npy'
file_path_5 = path_utils.get_plot_path() +'/point/corner1_hybrid.npy'
file_path_6 = path_utils.get_plot_path() +'/point/corner2_push.npy'
file_path_7 = path_utils.get_plot_path() +'/point/corner2_pull.npy'
file_path_8 = path_utils.get_plot_path() +'/point/corner2_hybrid.npy'
push_data = np.load(file_path_1, mmap_mode="r")
pull_data = np.load(file_path_2, mmap_mode="r")
push_data_c1 = np.load(file_path_3, mmap_mode="r")
pull_data_c1 = np.load(file_path_4, mmap_mode="r")
hybrid_data_c1 = np.load(file_path_5, mmap_mode="r")
push_data_c2 = np.load(file_path_6, mmap_mode="r")
pull_data_c2 = np.load(file_path_7, mmap_mode="r")
hybrid_data_c2 = np.load(file_path_8, mmap_mode="r")
n = int(push_data_c2.size/19)
# print(push_data)
print(n)

'''
The data is saved in this form (for every row):
[save_time, 
 save_robot_pos_x, save_robot_pos_y, save_robot_vel_x, save_robot_vel_y, 
 save_block_pos_x, save_block_pos_y, save_block_pos_z, 
 save_block_quat_1, save_block_quat_2, save_block_quat_3, save_block_quat_4, 
 block_goal_x, block_goal_y, 
 avg_sim_freq, avg_task_freq, avg_mot_freq, dyn_obs_coll, task_time]
'''

# Compute cost of distance and quaternion
def compute_cost(data_array, n):
    # orientation error between the block and the goal
    quat_cost = skill_utils.get_general_ori_cube2goal(torch.tensor(data_array[:, 8:12]), 
                                             torch.tensor([0, 0, 0, 1]).repeat(n).view(n, 4))
    # distance between the block and the goal
    dist_disp = data_array[:, 5:7] - data_array[:, 12:14]
    pos_cost = np.linalg.norm(dist_disp, axis=1)
    quat_cost = quat_cost.cpu().detach().numpy()
    return pos_cost, quat_cost

def compute_mean_std(data_array, string):
    print(string, format(np.mean(data_array), '.4f'), '±', format(np.std(data_array), '.4f'))

push_pos_cost, push_quat_cost = compute_cost(push_data, 60)
push_dyn_obs_coll = push_data[:, 17]
push_task_time = push_data[:, 18]
pull_pos_cost, pull_quat_cost = compute_cost(pull_data, 60)
pull_dyn_obs_coll = pull_data[:, 17]
pull_task_time = pull_data[:, 18]
push_c1_pos_cost, push_c1_quat_cost = compute_cost(push_data_c1, 20)
push_c1_task_time = push_data_c1[:, 18]
pull_c1_pos_cost, pull_c1_quat_cost = compute_cost(pull_data_c1, 20)
pull_c1_task_time = pull_data_c1[:, 18]
hybrid_c1_pos_cost, hybrid_c1_quat_cost = compute_cost(hybrid_data_c1, 20)
hybrid_c1_task_time = hybrid_data_c1[:, 18]
hybrid_c2_pos_cost, hybrid_c2_quat_cost = compute_cost(hybrid_data_c2, 20)
hybrid_c2_task_time = hybrid_data_c2[:, 18]
pull_c2_pos_cost, pull_c2_quat_cost = compute_cost(pull_data_c2, 20)
pull_c2_task_time = pull_data_c2[:, 18]
push_c2_pos_cost, push_c2_quat_cost = compute_cost(push_data_c2, 20)
push_c2_task_time = push_data_c2[:, 18]

# Compute mean and std
# print('---------Case 2 push---------')
# compute_mean_std(push_pos_cost, 'pos')
# compute_mean_std(push_quat_cost, 'quat')
# compute_mean_std(push_dyn_obs_coll, 'obs collision')
# compute_mean_std(push_task_time, 'task time')
# print('---------Case 2 pull---------')
# compute_mean_std(pull_pos_cost, 'pos')
# compute_mean_std(pull_quat_cost, 'quat')
# compute_mean_std(pull_dyn_obs_coll, 'obs collision')
# compute_mean_std(pull_task_time, 'task time')
print('---------Case 3 push---------')
compute_mean_std(push_c1_pos_cost, 'pos')
compute_mean_std(push_c1_quat_cost, 'quat')
compute_mean_std(push_c1_task_time, 'task time')
print('---------Case 3 pull---------')
compute_mean_std(pull_c1_pos_cost, 'pos')
compute_mean_std(pull_c1_quat_cost, 'quat')
compute_mean_std(pull_c1_task_time, 'task time')
print('---------Case 3 hybrid---------')
compute_mean_std(hybrid_c1_pos_cost, 'pos')
compute_mean_std(hybrid_c1_quat_cost, 'quat')
compute_mean_std(hybrid_c1_task_time, 'task time')
print('---------Case 4 push---------')
compute_mean_std(push_c2_pos_cost, 'pos')
compute_mean_std(push_c2_quat_cost, 'quat')
compute_mean_std(push_c2_task_time, 'task time')
print('---------Case 4 pull---------')
compute_mean_std(pull_c2_pos_cost, 'pos')
compute_mean_std(pull_c2_quat_cost, 'quat')
compute_mean_std(pull_c2_task_time, 'task time')
print('---------Case 4 hybrid---------')
compute_mean_std(hybrid_c2_pos_cost, 'pos')
compute_mean_std(hybrid_c2_quat_cost, 'quat')
compute_mean_std(hybrid_c2_task_time, 'task time')

# # Box plot
# fig = go.Figure()
# label_x = ['pos_error']*60+['ori_error']*60+['collisions']*60+['task time']*60
# fig.add_trace(go.Box(y=np.concatenate((push_pos_cost, push_quat_cost, push_dyn_obs_coll, push_task_time)), x=label_x, name="push", marker_color='#FF4136'))
# fig.add_trace(go.Box(y=np.concatenate((pull_pos_cost, pull_quat_cost, pull_dyn_obs_coll, pull_task_time)), x=label_x, name="pull", marker_color='#FF851B'))

# fig.update_layout(
#     title = 'Results of Normal Case',
#     title_x=0.5,
#     yaxis_title='Logarithmic scale',
#     boxmode='group', # group together boxes of the different traces for each value of x
#     boxgroupgap=0.3, # update
#     boxgap=0
# )
# fig.update_yaxes(type="log")
# fig.show() 

# Box plot
fig = go.Figure()
label_x = ['pos_error']*20+['ori_error']*20+['task time']*20
fig.add_trace(go.Box(y=np.concatenate((push_c1_pos_cost, push_c1_quat_cost, push_c1_task_time)), x=label_x, name="push", marker_color='#FF4136'))
fig.add_trace(go.Box(y=np.concatenate((pull_c1_pos_cost, pull_c1_quat_cost, pull_c1_task_time)), x=label_x, name="pull", marker_color='#FF851B'))
fig.add_trace(go.Box(y=np.concatenate((hybrid_c1_pos_cost, hybrid_c1_quat_cost, hybrid_c1_task_time)), x=label_x, name="hybrid", marker_color='#3D9970'))

fig.update_layout(
    title = 'Results of One Corner Case',
    title_x=0.5,
    yaxis_title='Logarithmic scale',
    boxmode='group', # group together boxes of the different traces for each value of x
    boxgroupgap=0.3, # update
    boxgap=0
)
fig.update_yaxes(type="log")
fig.show() 

# Box plot
fig = go.Figure()
fig.add_trace(go.Box(y=np.concatenate((push_c2_pos_cost, push_c2_quat_cost, push_c2_task_time)), x=label_x, name="push", marker_color='#FF4136'))
fig.add_trace(go.Box(y=np.concatenate((pull_c2_pos_cost, pull_c2_quat_cost, pull_c2_task_time)), x=label_x, name="pull", marker_color='#FF851B'))
fig.add_trace(go.Box(y=np.concatenate((hybrid_c2_pos_cost, hybrid_c2_quat_cost, hybrid_c2_task_time)), x=label_x, name="hybrid", marker_color='#3D9970'))

fig.update_layout(
    title = 'Results of Two Corners Case',
    title_x=0.5,
    yaxis_title='Logarithmic scale',
    boxmode='group', # group together boxes of the different traces for each value of x
    boxgroupgap=0.3, # update
    boxgap=0
)
fig.update_yaxes(type="log")
fig.show() 