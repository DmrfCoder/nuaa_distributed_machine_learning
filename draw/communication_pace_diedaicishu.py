import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import make_interp_spline


def shunhua(y):
    x = []
    index = 0
    for i in y:
        x.append(index)
        index += 1
    xnew = np.linspace(0, 10, 77)
    return make_interp_spline(x, y)(xnew)


base_path = '/Users/dmrfcoder/Documents/毕业设计/code/nuaa_distributed_machine_learning/'
improve_speed_shijian = np.loadtxt(base_path + 'improve_speed_x.txt')
improve_speed_zhunquelv = np.loadtxt(base_path + 'improve_speed_y.txt')
improve_speed_diedaicishu = np.loadtxt(base_path + 'improve_speed_plt_iter.txt')

BSP_shijian = np.loadtxt(base_path + 'bsp_x.txt')
BSP_zhunquelv = np.loadtxt(base_path + 'bsp_y.txt')
BSP_diedaicishu = np.loadtxt(base_path + 'bsp_plt_iter.txt')


index = len(BSP_shijian)

# plt.ylabel('accuracy')
# plt.xlabel('iterations')
# plt.plot(improve_speed_diedaicishu, improve_speed_zhunquelv, label=u'BSP通信步调')
#
# plt.plot(BSP_diedaicishu[0:index], BSP_zhunquelv[0:index], color='red', label=u'本文设计的通信步调')



improve_x = []
improve_y = []

for index2 in range(len(improve_speed_zhunquelv)):
    value = BSP_zhunquelv[index2] - improve_speed_zhunquelv[index2]
    value = value / improve_speed_zhunquelv[index2]
    improve_y.append(value)
improve_y=shunhua(improve_y)
plt.plot(improve_speed_diedaicishu,improve_y)

plt.legend()
plt.show()
