import time

import numpy as np
import ray
import torch

from mnist.Net import ConvNet


@ray.remote
class ParameterServer(object):
    def __init__(self, lr):
        self.model = ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.time_costs = []

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def apply_gradients2(self, start_time, *gradients):
        self.time_costs.append(time.time() - start_time)
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        result_time_start = time.time()
        return self.model.get_weights(), result_time_start

    def get_weights(self):
        result_time_start = time.time()

        return self.model.get_weights(), result_time_start

    def get_cost_time(self):
        return self.time_costs
