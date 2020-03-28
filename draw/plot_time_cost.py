import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('/Users/dmrfcoder/Documents/毕业设计/code/nuaa_distributed_machine_learning/worker_to_bsp.txt')
bsp_to_worker_time_start = np.loadtxt(
    '/Users/dmrfcoder/Documents/毕业设计/code/nuaa_distributed_machine_learning/bsp_to_worker_time_start.txt')
bsp_to_worker_time_end = np.loadtxt(
    '/Users/dmrfcoder/Documents/毕业设计/code/nuaa_distributed_machine_learning/bsp_to_worker_time_end.txt')
cost = bsp_to_worker_time_end - bsp_to_worker_time_start
plt.plot(data, color='red')
#plt.plot(bsp_to_worker_time_end, color='black', label='bsp_to_worker_time_end')
#plt.plot(cost, color='blue', label='cost')

plt.show()
