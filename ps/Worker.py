import time
from collections import namedtuple

import ray
import torch.nn.functional as F

from mnist.Net import ConvNet

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])
import torch


@ray.remote
class Worker(object):
    def __init__(self, id, sleep_time, data_iterator, iterations=200, should_compute_count=5):
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
        self.should_compute_count = should_compute_count

        print('init worker_' + str(id) + 'with sleep_time:' + str(sleep_time))

    def compute_gradients(self, weights, start_time, need_quantize=False, improve_speed=False):
        current_time = time.time()
        self.time_start.append(start_time)
        self.time_costs.append(current_time - start_time)
        self.time_stamp.append(current_time)
        self.time_end.append(current_time)
        # print('接收到参数,耗时：',current_time - start_time)

        # 下面的代码用于异步情况下控制当前节点的训练次数
        # if self.computed_count == self.iterations:
        #     return None
        if weights is not None:  # 如果weights为None即表示本次不进行通信，使用旧的本地模型继续进行计算
            self.model.set_weights(weights)
        current_compute_count = 0
        for index in range(self.should_compute_count):
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
            current_compute_count += 1
            if self.sleep_time > 0:
                time.sleep(self.sleep_time)
            if not improve_speed:
                break

        gradients = self.model.get_gradients()
        if need_quantize:
            quantized_gradient = []
            for item_gradient in gradients:
                quantized_gradient.append(self.quantize_tensor(torch.tensor(item_gradient, dtype=torch.float)))
            start_time = time.time()
            return quantized_gradient, start_time, current_compute_count  # 通信的内容
        start_time = time.time()
        return gradients, start_time, current_compute_count  # 通信的内容

    def computing_power_test(self):
        return self.sleep_time

    def get_cost_time(self):
        return self.time_costs, self.time_stamp, self.time_start, self.time_end

    def quantize_tensor(self, x, num_bits=8):
        qmin = 0.
        qmax = 2. ** num_bits - 1.
        min_val, max_val = x.min(), x.max()

        scale = (max_val - min_val) / (qmax - qmin)

        initial_zero_point = qmin - min_val / scale

        zero_point = 0
        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        else:
            zero_point = initial_zero_point

        zero_point = int(zero_point)
        q_x = zero_point + x / scale
        q_x.clamp_(qmin, qmax).round_()
        q_x = q_x.round().byte()
        # print('量化：',q_x)
        return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

    def dequantize_tensor(self, q_x):
        return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

