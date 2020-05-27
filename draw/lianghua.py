import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('quantization@accuracy.pdf')
plt.figure(figsize=(8, 7))


base_path = '/Users/dmrfcoder/Documents/毕业设计/code/nuaa_distributed_machine_learning/'
lianghuaqian_x = np.loadtxt(base_path + 'lianghua_x.txt')
lianghuaqian_y = np.loadtxt(base_path + 'lianghua_y.txt')
lianghuahou_x = np.loadtxt(base_path + 'lianghuahou_x.txt')
lianghuahou_y = np.loadtxt(base_path + 'lianghuahou_y.txt')

fsize = 20


plt.plot(lianghuaqian_x, lianghuaqian_y, label='before quantization')
plt.plot(lianghuahou_x, lianghuahou_y, label='after quantization')
plt.ylabel('accuracy', fontsize=fsize)
plt.xlabel('iteration', fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.legend(fontsize=fsize)
pp.savefig()
plt.close()
pp.close()
