import time

import ray


@ray.remote
class Server:
    def demo(self, data):
        print('接收到数据：' + str(time.time()))
        start_time = time.time()
        return data, start_time
