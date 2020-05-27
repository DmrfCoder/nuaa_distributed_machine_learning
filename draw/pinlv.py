import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('frequency@adjustment@accuracy.pdf')

plt.figure(figsize=(9, 9))

base_path = '/Users/dmrfcoder/Documents/毕业设计/code/nuaa_distributed_machine_learning/'
pinlvhou_x = np.loadtxt(base_path + 'pinlvhou_x.txt')
pinlvhou_y = np.loadtxt(base_path + 'pinlvhou_y.txt')
pinlvqian_x = np.loadtxt(base_path + 'pinlvqian_x.txt')
pinlvqian_y = np.loadtxt(base_path + 'pinlvqian_y.txt')

index1 = 0
for item in pinlvqian_x:
    if item >= 100:
        break
    index1 += 1

index2 = 0

for item in pinlvhou_x:
    if item >= 100:
        break
    index2 += 1
fsize = 20

plt.plot(pinlvqian_x[0:index1], pinlvqian_y[0:index1], label='before frequency adjustment')
plt.plot(pinlvhou_x[0:index2], pinlvhou_y[0:index2], label='after frequency adjustment')
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.ylabel('accuracy', fontsize=fsize)
plt.xlabel('iteration', fontsize=fsize)
plt.legend(fontsize=fsize)
pp.savefig()
plt.close()
pp.close()
