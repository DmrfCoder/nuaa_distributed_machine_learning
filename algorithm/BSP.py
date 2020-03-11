import ray
import torch
from matplotlib import pyplot as plt

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

        print("Running synchronous parameter server training.")
        # 全局模型当前的参数
        current_weights = ps.get_weights.remote()
        for i in range(self.iterations):
            # gradients是一个ObjectID的列表，这些ObjectID对应的任务是worker.compute_gradients，传入的参数是全局模型的参数，返回值是本地worker计算后的本地模型的参数值
            gradients = [
                worker.compute_gradients.remote(current_weights) for worker in workers
            ]

            # current_weights是ps.apply_gradients这个异步任务对应的ObjectID，传入的参数是workers训练完毕的参数，返回值是参数服务器更新后的最新的模型
            # Calculate update after all gradients are available.
            current_weights = ps.apply_gradients.remote(*gradients)

            if i % 10 == 0:
                # Evaluate the current model.
                # 将workers中的参数计算出来然后设置获取当前参数服务器上全局模型的参数
                model.set_weights(ray.get(current_weights))
                accuracy = self.evaluate(model, test_loader)
                plt_x.append(i)
                plt_y.append(accuracy)
                print("Iter {}: \taccuracy is {:.1f}".format(i * self.num_workers, accuracy))
        print("Final accuracy is {:.1f}.".format(accuracy))
        # Clean up Ray resources and processes before the next example.
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
