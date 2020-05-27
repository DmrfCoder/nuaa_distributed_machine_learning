import time

import ray
import torch
from matplotlib import pyplot as plt

from mnist.Net import ConvNet
from ps.Server import ParameterServer
from ps.Worker import Worker


class ASP:
    def __init__(self, sleep_gap, data_loaders, num_workers, dataset_factory, iterations, improve_data=False):
        self.sleep_gap = sleep_gap
        self.data_loaders = data_loaders
        self.num_workers = num_workers
        self.dataset_factory = dataset_factory
        self.iterations = iterations
        self.improve_data = improve_data

    def execute(self, show_plt=True, need_quantize=False, improve_speed=False):
        ###########################################################################
        # Asynchronous Parameter Server Training
        # --------------------------------------
        # We'll now create a synchronous parameter server training scheme. We'll first
        # instantiate a process for the parameter server, along with multiple
        # workers.

        print("Running Asynchronous Parameter Server Training.")

        ray.init(ignore_reinit_error=True)
        ps = ParameterServer.remote(1e-2)
        if improve_speed:
            cumputing_powper = [(i + 1) * self.sleep_gap for i in range(self.num_workers)]
            self.data_loaders = self.dataset_factory.build_dataset_with_power(cumputing_powper, shuffle=True)
        workers = [Worker.remote(i, (i + 1) * self.sleep_gap, self.data_loaders[i], (self.num_workers / i))
                   for i in range(self.num_workers)]
        model = ConvNet()
        test_loader = self.dataset_factory.get_test_loader()
        ###########################################################################
        # Here, workers will asynchronously compute the gradients given its
        # current weights and send these gradients to the parameter server as
        # soon as they are ready. When the Parameter server finishes applying the
        # new gradient, the server will send back a copy of the current weights to the
        # worker. The worker will then update the weights and repeat.
        current_weights = ps.get_weights.remote()

        gradients = {}
        for worker in workers:
            # 将ps上的参数发送给worker，即获取ps上的参数，即pull,代码层面的通信流是Server--->ASP--->Worker
            current_time = time.time()
            w, result_time_start = ray.get(current_weights)
            gradients[worker.compute_gradients.remote(w, current_time, need_quantize,
                                                      improve_speed)] = worker
        '''
        gradients={ObjectID(7e0a4dfc4c87306fef0a6c22010000c801000000): Actor(Worker, ef0a6c220100), ObjectID(6f53dca1f451ca94f66d17ba010000c801000000): Actor(Worker, f66d17ba0100)}
        list(gradients)=[ObjectID(7e0a4dfc4c87306fef0a6c22010000c801000000), ObjectID(6f53dca1f451ca94f66d17ba010000c801000000)]
        一个循环一次只会计算一个worker的compute_gradients，一共有
        '''
        i = 0
        plt_x = []
        plt_y = []
        start_train_time = time.time()
        while True:
            '''
            wait(object_ids, num_returns=1, timeout=None),这里传入了所有worker的compute_gradients的任务，
            而num_returns默认是1，即只要有1个worker完成了compute_gradients，ray.wait(list(gradients))就会返回，
            ready_gradient_list是完成的数据对象, _ 是未完成的ObjectID
            '''
            if len(list(gradients)) == 0:
                break
            ready_gradient_list, _ = ray.wait(list(gradients))
            # 拿到第一个完成的worker的Objectid（事实上只有1个）
            ready_gradient_id = ready_gradient_list[0]
            '''
            返回的worker是计算完成的那个worker，并且计算完成的<ObjectId,worker>会从gradients中删除
            '''
            worker = gradients.pop(ready_gradient_id)
            # 这里相当于是每次worker计算完成之后都进行通信
            # 将完成的这个worker的计算结果上传到ps，并得到ps更新后的模型,即进行通信，执行push操作,所以代码层面的通信流是Worker--->ASP--->Server
            if ray.get(ready_gradient_id) is not None:
                temp_gradients = []
                item_gradient, start_time, current_compute_count = ray.get(ready_gradient_id)
                temp_gradients.append(item_gradient)
                current_weights = ps.apply_gradients.remote(*temp_gradients)
                current_time = time.time()
                weights = ray.get(current_weights)

                gradients[worker.compute_gradients.remote(weights, current_time, need_quantize,
                                                          improve_speed)] = worker

            if i % 10 == 0:
                # Evaluate the current model after every 10 updates.
                model.set_weights(ray.get(current_weights))
                accuracy = self.evaluate(model, test_loader)
                # plt_x.append(i)
                plt_x.append(time.time() - start_train_time)
                plt_y.append(accuracy)
                print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))
            i += 1
            if i >= self.iterations:
                break

        print("Final accuracy is {:.1f}.".format(accuracy))
        ray.shutdown()
        if show_plt:
            plt.title('ASP-numWorkers:' + str(self.num_workers) + 'sleepGap:' + str(self.sleep_gap))
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
