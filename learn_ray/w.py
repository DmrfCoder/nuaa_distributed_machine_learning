import time

import ray


@ray.remote
class Worker:
    def demo(self, index=0, start_time=0):
        print('index:' + str(index) + ' 被执行,传输耗时：' + str(time.time() - start_time))
        return 0
