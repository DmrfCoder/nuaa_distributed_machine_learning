import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline

y=np.loadtxt('/Users/dmrfcoder/Documents/毕业设计/code/nuaa_distributed_machine_learning/demo.txt')
print(y)

g=[]
x=[]
# index=0
# gap=20
# x_index=0
# while True:
#     if index+gap>=len(y):
#         break
#     g.append((y[index+gap]-y[index])/gap)
#     index+=gap
#     x.append(x_index)
#     x_index+=1
index=0
for i in y:
    x.append(index)
    index+=1

# g=y
# xnew = np.linspace(0,index,30)
# power_smooth = make_interp_spline(x,y)(xnew)

plt.plot(y)
plt.show()