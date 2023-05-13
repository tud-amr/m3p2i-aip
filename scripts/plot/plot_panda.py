import os, sys, time
sys.path.append('../')
from utils import path_utils
from npy_append_array import NpyAppendArray
import numpy as np
import matplotlib.pyplot as plt

file_path_1 = path_utils.get_plot_path() +'/panda/normal_pick.npy'
file_path_2 = path_utils.get_plot_path() +'/panda/reactive_pick.npy'
tamp_normal_data = np.load(file_path_1, mmap_mode="r")
tamp_reactive_data = np.load(file_path_2, mmap_mode="r")

print(tamp_normal_data.size)
print(tamp_reactive_data.size)
# print(data[-1, 0])
# print(time.asctime(time.localtime(data[-1, 0])))

# Compute distance
def plot_pos_disp(data_array, string):
    n = int(data_array.size/15)
    dist_disp = data_array[:, 1:3] - data_array[:, 8:10]
    dist_norm = np.linalg.norm(dist_disp, axis=1)
    quat_disp = data_array[:, 4:8] - data_array[:, 11:]
    quat_norm = np.linalg.norm(quat_disp, axis=1)
    print('dist norm', dist_norm)
    print('quat norm', quat_norm)
    plt.scatter(data_array[:, 1], data_array[:, 2], c='red')
    plt.scatter(data_array[:, 8], data_array[:, 9], c='green')
    for i in range(n):
        plt.plot([data_array[i, 1], data_array[i, 8]], [data_array[i, 2], data_array[i, 9]], 'k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Position displacement (' + string + ' task)')
    plt.show()

plot_pos_disp(tamp_normal_data, 'normal')
# plot_pos_disp(tamp_reactive_data, 'reactive')