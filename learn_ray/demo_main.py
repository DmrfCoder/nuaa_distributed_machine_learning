import time

import ray

from learn_ray.s import Server
from learn_ray.w import Worker

ray.init(ignore_reinit_error=True)
s = Server.remote()
w1 = Worker.remote()
w2 = Worker.remote()
w3 = Worker.remote()
w4 = Worker.remote()
w5 = Worker.remote()
w6 = Worker.remote()
workers = [w1, w2, w3, w4, w5, w6]
# workers[2].demo.remote(2,time.time())
gradients_id = []
for i in range(1000):
    time.sleep(0.5)

    for i in range(6):
        gradients_id.append(workers[i].demo.remote(i, time.time()))
# ray.get(gradients_id[3])
ray.shutdown()
'''
- 如果get的时候task已经完成，start_time应该为当前时间，耗时为a-->b的传输时间
- 如果get的时候task没有完成，start_time应该为task中传出的时间
'''
