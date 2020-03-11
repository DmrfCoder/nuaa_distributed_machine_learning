import ray
import torch
from matplotlib import pyplot as plt

from mnist.Net import ConvNet
from ps.Server import ParameterServer
from ps.Worker import Worker


class SSP:
    def __init__(self, sleep_gap, data_loaders, num_workers, dataset_factory, iterations, min_lock):
        self.sleep_gap = sleep_gap
        self.data_loaders = data_loaders
        self.num_workers = num_workers
        self.dataset_factory = dataset_factory
        self.iterations = iterations
        self.min_lock = min_lock

    def execute(self, show_plt=True):
        plt_x = []
        plt_y = []
        ray.init(ignore_reinit_error=True)
        ps = ParameterServer.remote(1e-2)
        workers = [Worker.remote(i, (i + 1) * self.sleep_gap, self.data_loaders[i]) for i in
                   range(self.num_workers)]
        model = ConvNet()
        test_loader = self.dataset_factory.get_test_loader()
        current_weights = ps.get_weights.remote()

        gradients = {}
        waiting_gradients = []
        worker_index = {}
        computed_count = []
        start_index = 0
        for worker in workers:
            gradients[worker.compute_gradients.remote(current_weights)] = worker
            worker_index[worker] = start_index
            computed_count.append(start_index)
            start_index += 1

        i = 0

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

            index = worker_index[worker]
            computed_count[index] += 1
            for item in waiting_gradients:
                if computed_count[worker_index[item]] - min(computed_count) < self.min_lock:
                    if ray.get(ready_gradient_id) is not None:
                        gradients[item.compute_gradients.remote(current_weights)] = item
                    # print('remove worker_'+str(worker_index[item])+' from waiting_gradients')
                    waiting_gradients.remove(item)

            # 将完成的这个worker的计算结果上传到ps，并得到ps更新后的模型
            if computed_count[index] - min(computed_count) >= self.min_lock:
                if worker not in waiting_gradients:
                    # print('put worker_'+str(index)+' in waiting_gradients')
                    waiting_gradients.append(worker)
            elif ray.get(ready_gradient_id) is not None:
                current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
                gradients[worker.compute_gradients.remote(current_weights)] = worker

            if i % 10 == 0:
                # Evaluate the current model after every 10 updates.
                model.set_weights(ray.get(current_weights))
                accuracy = self.evaluate(model, test_loader)
                print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))
                plt_x.append(i)
                plt_y.append(accuracy)
            i += 1
        print("Final accuracy is {:.1f}.".format(accuracy))
        ray.shutdown()
        if show_plt:
            plt.title(
                'SSP-numWorkers:' + str(self.num_workers) + 'sleepGap:' + str(self.sleep_gap) + ' min_lock:' + str(
                    self.min_lock))
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
