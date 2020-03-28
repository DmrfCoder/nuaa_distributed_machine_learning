import time

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch

from mnist.Net import ConvNet
from ps.Server import ParameterServer
from ps.Worker import Worker


class BSP:
    def __init__(self, sleep_gap, data_loaders, num_workers, dataset_factory, iterations):
        self.sleep_gap = sleep_gap
        self.data_loaders = data_loaders
        self.num_workers = num_workers
        self.dataset_factory = dataset_factory
        self.iterations = iterations

    def execute(self, show_plt=True):
        ###########################################################################
        # Synchronous Parameter Server Training
        # -------------------------------------
        # We'll now create a synchronous parameter server training scheme. We'll first
        # instantiate a process for the parameter server, along with multiple
        # workers.
        plt_x = []
        plt_y = []
        ray.init(ignore_reinit_error=True)
        '''
        初始化ps和workers
        '''
        ps = ParameterServer.remote(1e-2)
        workers = [Worker.remote(i, (i + 1) * self.sleep_gap, self.data_loaders[i]) for i in range(self.num_workers)]

        ###########################################################################
        # We'll also instantiate a model on the driver process to evaluate the test
        # accuracy during training.
        # 初始化全局model和test_loader
        model = ConvNet()
        test_loader = self.dataset_factory.get_test_loader()

        ###########################################################################
        # Training alternates between:
        #
        # 1. Computing the gradients given the current weights from the server
        # 2. Updating the parameter server's weights with the gradients.

        print("Running BSP parameter server training.")
        # 全局模型当前的参数
        current_weights = ps.get_weights.remote()
        worker_to_bsp = []
        server_to_bsp = []
        pre_acc = 0.0
        acc_gap = 0.0
        for i in range(self.iterations):
            # gradients是一个ObjectID的列表，这些ObjectID对应的任务是worker.compute_gradients，传入的参数是全局模型的参数，返回值是本地worker计算后的本地模型的参数值
            '''
            server-->bsp-->worker
            '''
            weights = None
            if current_weights is not None:
                start_get_time = time.time()
                weights, result_time_start = ray.get(current_weights)
                if start_get_time > result_time_start:  # 如果调用get之前task已经完成了，那么通信起始时间应该从调用get的时候算
                    cost_time = time.time() - start_get_time
                else:  # 如果调用get之前task还没有完成，那么通信起始时间应该从task最后return之前算
                    cost_time = time.time() - result_time_start

                server_to_bsp.append(cost_time)

            gradients_id = [
                # worker.compute_gradients.remote(weights, time.time()) for worker in workers
            ]

            for index in range(self.num_workers):
                current_time = time.time()
                gradients_id_item = workers[index].compute_gradients.remote(weights, current_time)
                gradients_id.append(gradients_id_item)

            # current_weights是ps.apply_gradients这个异步任务对应的ObjectID，传入的参数是workers训练完毕的参数，返回值是参数服务器更新后的最新的模型
            # Calculate update after all gradients are available.
            # if i % 10 == 0:
            gradients = []
            for id in gradients_id:
                '''
                worker--->bsp
                '''
                # start_time是worker启动数据传输的时间
                start_get_time = time.time()
                item_gradient, result_time_start = ray.get(id)

                if start_get_time > result_time_start:  # 如果调用get之前task已经完成了，那么通信起始时间应该从调用get的时候算
                    cost_time = time.time() - start_get_time
                else:  # 如果调用get之前task还没有完成，那么通信起始时间应该从task最后return之前算
                    cost_time = time.time() - result_time_start

                worker_to_bsp.append(cost_time)
                gradients.append(item_gradient)
            # bsp-->server
            '''
            bsp-->server
            '''
            current_weights = ps.apply_gradients2.remote(time.time(), *gradients)
            # else:
            #     current_weights = None

            if (i * self.num_workers) % 10 == 0:
                # Evaluate the current model.
                # 将workers中的参数计算出来然后设置获取当前参数服务器上全局模型的参数
                # 通信：server-->BSP
                '''
                server-->bsp
                '''
                start_get_time = time.time()
                weights, result_time_start = ray.get(current_weights)
                if start_get_time > result_time_start:  # 如果调用get之前task已经完成了，那么通信起始时间应该从调用get的时候算
                    cost_time = time.time() - start_get_time
                else:  # 如果调用get之前task还没有完成，那么通信起始时间应该从task最后return之前算
                    cost_time = time.time() - result_time_start

                server_to_bsp.append(cost_time)

                model.set_weights(weights)
                accuracy = self.evaluate(model, test_loader)
                plt_x.append(i)
                plt_y.append(accuracy)
                if pre_acc == 0.0:
                    pre_acc = accuracy
                elif acc_gap == 0.0:
                    acc_gap = accuracy - pre_acc
                    pre_acc = accuracy
                elif accuracy - pre_acc < acc_gap:
                    acc_gap = accuracy - pre_acc
                    pre_acc = accuracy
                    # 减少通信次数
                    pass
                else:
                    acc_gap = accuracy - pre_acc
                    pre_acc = accuracy
                    # 增多通信次数
                    pass
                print("Iter {}: \taccuracy is {:.1f}".format(i * self.num_workers, accuracy))
        print("Final accuracy is {:.1f}.".format(accuracy))
        # Clean up Ray resources and processes before the next example.
        bsp_to_worker = []
        bsp_to_worker_time_stamp = []
        bsp_to_worker_time_start = []
        bsp_to_worker_time_end = []
        for worker in workers:
            cost_time, time_stamp, time_start, time_end = ray.get(worker.get_cost_time.remote())
            bsp_to_worker.extend(cost_time)
            bsp_to_worker_time_stamp.extend(time_stamp)
            bsp_to_worker_time_start.extend(time_start)
            bsp_to_worker_time_end.extend(time_end)
        Z = zip(bsp_to_worker_time_stamp, bsp_to_worker, bsp_to_worker_time_start, bsp_to_worker_time_end)
        Z = sorted(Z, reverse=False)
        bsp_to_worker_time_stamp, bsp_to_worker, bsp_to_worker_time_start, bsp_to_worker_time_end = zip(*Z)
        bsp_to_server = ray.get(ps.get_cost_time.remote())

        # print(bsp_to_worker)
        # print(bsp_to_server)
        # print(worker_to_bsp)
        # print(server_to_bsp)
        sum_bsp_to_worker = sum(bsp_to_worker)
        sum_bsp_to_server = sum(bsp_to_server)
        sum_worker_to_bsp = sum(worker_to_bsp)
        sum_server_to_bsp = sum(server_to_bsp)
        print('sum_bsp_to_worker:', sum_bsp_to_worker)
        print('sum_bsp_to_server:', sum_bsp_to_server)
        print('sum_worker_to_bsp:', sum_worker_to_bsp)
        print('sum_server_to_bsp:', sum_server_to_bsp)
        print('通信总耗时：' + str(sum_bsp_to_server + sum_bsp_to_worker + sum_worker_to_bsp + sum_server_to_bsp) + ' s')

        bsp_to_worker = np.array(bsp_to_worker)
        bsp_to_server = np.array(bsp_to_server)
        worker_to_bsp = np.array(worker_to_bsp)
        server_to_bsp = np.array(server_to_bsp)
        bsp_to_worker_time_start = np.array(bsp_to_worker_time_start)
        bsp_to_worker_time_end = np.array(bsp_to_worker_time_end)
        np.savetxt('bsp_to_worker.txt', bsp_to_worker)
        np.savetxt('bsp_to_worker_time_start.txt', bsp_to_worker_time_start)
        np.savetxt('bsp_to_worker_time_end.txt', bsp_to_worker_time_end)
        # plt.plot(bsp_to_worker, color='red', label='bsp_to_worker')

        # plt.plot(bsp_to_worker_time_start, color='black', label='bsp_to_worker_time_start')
        # plt.show()
        np.savetxt('bsp_to_worker.txt', bsp_to_worker)
        np.savetxt('bsp_to_server.txt', bsp_to_server)
        np.savetxt('worker_to_bsp.txt', worker_to_bsp)
        np.savetxt('server_to_bsp.txt', server_to_bsp)
        ray.shutdown()

        if show_plt:
            plt.title('BSP-numWorkers:' + str(self.num_workers) + 'sleepGap:' + str(self.sleep_gap))
            plt.plot(plt_x, plt_y)
            plt.show()

    def evaluate(self, model, test_loader):
        """在验证集（validation dataset）上测试模型的准确率"""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                # This is only set to finish evaluation faster.
                if batch_idx * len(data) > 1024:
                    break
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100. * correct / total
