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
        print('init worker_' + str(id) + 'with sleep_time:' + str(sleep_time))

    def compute_gradients(self, weights):
        if self.computed_count == self.iterations:
            return None
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

        return self.model.get_gradients()  # 通信的内容

    def computing_power_test(self):
        return self.sleep_time
