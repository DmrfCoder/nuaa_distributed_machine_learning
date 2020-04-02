import time

import ray
import torch.nn.functional as F

from mnist.Net import ConvNet


@ray.remote
class Worker(object):
    def __init__(self, id, sleep_time, data_iterator, iterations=200):
        self.model = ConvNet()
        self.work_id = id
        self.init_data_iterator = data_iterator
        self.data_iterator = iter(self.init_data_iterator)
        self.computed_count = 0
        self.sleep_time = sleep_time  # 单位s
        self.iterations = iterations
        self.time_costs = []
        self.time_stamp = []
        self.time_start = []
        self.time_end = []
        self.remaining_times = 1

        print('init worker_' + str(id) + 'with sleep_time:' + str(sleep_time))

    def compute_gradients(self, weights, start_time):
        current_time = time.time()
        self.time_start.append(start_time)
        self.time_costs.append(current_time - start_time)
        self.time_stamp.append(current_time)
        self.time_end.append(current_time)
        #print('接收到参数,耗时：',current_time - start_time)

        if self.computed_count == self.iterations:
            return None
        if weights is not None:  # 如果weights为None即表示本次不进行通信，使用旧的本地模型继续进行计算
            self.model.set_weights(weights)
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(self.init_data_iterator)
            data, target = next(self.data_iterator)

        self.model.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.computed_count += 1
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
        start_time = time.time()
        return self.model.get_gradients(), start_time  # 通信的内容

    def computing_power_test(self):
        return self.sleep_time

    def get_cost_time(self):
        return self.time_costs, self.time_stamp, self.time_start, self.time_end
