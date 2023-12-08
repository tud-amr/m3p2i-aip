import os, sys, time
sys.path.append('../')
from utils import path_utils
from npy_append_array import NpyAppendArray
import numpy as np
import torch

file_path = path_utils.get_plot_path() +'/panda/normal.npy'
for i in range(3):
    with NpyAppendArray(file_path) as npaa:
        curr_time = np.array([time.time()])
        a = torch.zeros(5, device="cuda:0")
        b = torch.zeros(3, device="cuda:0")
        np_arr_a = a.cpu().detach().numpy()
        np_arr_b = b.cpu().detach().numpy()
        concatenate_array = np.concatenate((curr_time, np_arr_a, np_arr_b))
        # print(np.concatenate((curr_time, np_arr_a, np_arr_b)))
        npaa.append(np.array([concatenate_array]))

data = np.load(file_path, mmap_mode="r")
# print(time.asctime( time.localtime(1683979648)))

print(data)
print(data[-1, 0])
print(time.asctime(time.localtime(data[-1, 0])))