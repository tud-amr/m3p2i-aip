import os, sys, time
sys.path.append('../')
from utils import path_utils, skill_utils
from npy_append_array import NpyAppendArray
import numpy as np
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go

file_path_1 = path_utils.get_plot_path() +'/point/case2_halton_push_coll.npy'
file_path_2 = path_utils.get_plot_path() +'/point/case2_halton_pull_coll.npy'
push_data = np.load(file_path_1, mmap_mode="r")
pull_data = np.load(file_path_2, mmap_mode="r")
n = int(pull_data.size/19)
# print(push_data)
print(n)

# Compute cost of distance and quaternion
def compute_cost(data_array):
    quat_cost = skill_utils.get_general_ori_cube2goal(torch.tensor(data_array[:, 8:12]), 
                                             torch.tensor([0, 0, 0, 1]).repeat(n).view(n, 4))
    dist_disp = data_array[:, 1:3] - data_array[:, 12:14]
    pos_cost = np.linalg.norm(dist_disp, axis=1)
    quat_cost = quat_cost.cpu().detach().numpy()
    return pos_cost, quat_cost

push_pos_cost, push_quat_cost = compute_cost(push_data)
push_dyn_obs_coll = push_data[:, 17]
push_task_time = push_data[:, 18]
pull_pos_cost, pull_quat_cost = compute_cost(pull_data)
pull_dyn_obs_coll = pull_data[:, 17]
pull_task_time = pull_data[:, 18]

# Box plot
fig = go.Figure()
label_x = ['pos_error']*60+['ori_error']*60+['collisions']*60+['task time']*60
fig.add_trace(go.Box(y=np.concatenate((push_pos_cost, push_quat_cost, push_dyn_obs_coll, push_task_time)), x=label_x, name="push", marker_color='#FF4136'))
fig.add_trace(go.Box(y=np.concatenate((pull_pos_cost, pull_quat_cost, pull_dyn_obs_coll, pull_task_time)), x=label_x, name="pull", marker_color='#FF851B'))

fig.update_layout(
    title = 'Metrics',
    title_x=0.5,
    # yaxis_title='Displacements',
    boxmode='group', # group together boxes of the different traces for each value of x
    boxgroupgap=0.3, # update
    boxgap=0
)
fig.show() 