from m3p2i_aip.utils import path_utils, skill_utils
import numpy as np
import torch
import plotly.graph_objects as go

file_path_1 = path_utils.get_plot_path() +'/panda/normal_pick.npy'
file_path_2 = path_utils.get_plot_path() +'/panda/reactive_pick.npy'
file_path_3 = path_utils.get_plot_path() +'/panda/normal_rl_pick_1500_64.npy'
file_path_4 = path_utils.get_plot_path() +'/panda/reactive_rl_pick_1500_64.npy'
tamp_normal_data = np.load(file_path_1, mmap_mode="r")
tamp_reactive_data = np.load(file_path_2, mmap_mode="r")
rl_normal_data = np.load(file_path_3, mmap_mode="r")
rl_normal_data = rl_normal_data[:50, :]
rl_reactive_data = np.load(file_path_4, mmap_mode="r")
rl_reactive_data = rl_reactive_data[:50, :]

print(np.shape(rl_normal_data))
print(rl_reactive_data.size)
# print(data[-1, 0])
# print(time.asctime(time.localtime(data[-1, 0])))

# Compute cost of distance and quaternion
def compute_cost(data_array):
    quat_cost = skill_utils.get_general_ori_cube2goal(torch.tensor(data_array[:, 4:8]), 
                                             torch.tensor(data_array[:, 11:]))
    dist_disp = data_array[:, 1:3] - data_array[:, 8:10]
    pos_cost = np.linalg.norm(dist_disp, axis=1)
    quat_cost = quat_cost.cpu().detach().numpy()
    return pos_cost, quat_cost

def compute_mean_std(data_array, string):
    print(string, format(np.mean(data_array), '.4f'), '±', format(np.std(data_array), '.4f'))

tamp_normal_pos, tamp_normal_quat = compute_cost(tamp_normal_data)
tamp_reactive_pos, tamp_reactive_quat = compute_cost(tamp_reactive_data)
rl_normal_pos, rl_normal_quat = compute_cost(rl_normal_data)
rl_reactive_pos, rl_reactive_quat = compute_cost(rl_reactive_data)

# Compute mean and std
print('---------TAMP normal---------')
compute_mean_std(tamp_normal_pos, 'pos')
compute_mean_std(tamp_normal_quat, 'quat')
print('---------RL normal---------')
compute_mean_std(rl_normal_pos, 'pos')
compute_mean_std(rl_normal_quat, 'quat')
print('---------TAMP reactive---------')
compute_mean_std(tamp_reactive_pos, 'pos')
compute_mean_std(tamp_reactive_quat, 'quat')
print('---------RL reactive---------')
compute_mean_std(rl_reactive_pos, 'pos')
compute_mean_std(rl_reactive_quat, 'quat')

# Box plot
fig = go.Figure()
tamp_x = ['Normal']*50+['Reactive']*50
rl_x = ['Normal']*64+['Reactive']*64
fig.add_trace(go.Box(y=np.concatenate((tamp_normal_pos, tamp_reactive_pos)), x=tamp_x, name="TAMP", marker_color='#FF4136'))
fig.add_trace(go.Box(y=np.concatenate((rl_normal_pos, rl_reactive_pos)), x=rl_x, name="RL", marker_color='#FF851B'))

fig.update_layout(
    title = 'Position error',
    title_x=0.5,
    yaxis_title='Logarithmic scale',
    boxmode='group', # group together boxes of the different traces for each value of x
    boxgroupgap=0.3, # update
    boxgap=0
)
fig.update_yaxes(type="log")
fig.show() 

fig = go.Figure()
fig.add_trace(go.Box(y=np.concatenate((tamp_normal_quat, tamp_reactive_quat)), x=tamp_x, name="TAMP", marker_color='#FF4136'))
fig.add_trace(go.Box(y=np.concatenate((rl_normal_quat, rl_reactive_quat)), x=rl_x, name="RL", marker_color='#FF851B'))

fig.update_layout(
    title = 'Orientation error',
    title_x=0.5,
    yaxis_title='Logarithmic scale',
    boxmode='group', # group together boxes of the different traces for each value of x
    boxgroupgap=0.3, # update
    boxgap=0
)
fig.update_yaxes(type="log")
fig.show() 