import os
import random

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


class CustomDataset(data.Dataset):
    def __init__(self, data, labels, transform=None,
                 target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data  # training set or test set
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class DatasetFactory:
    def __init__(self, data_root):
        self.root = data_root
    #如果shuffle为true表示数据为非独立同分布，如果shuffle为false表示数据为独立同分布
    def build_dataset(self, num_worker, shuffle=False):
        print('build data')
        mnist_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        training_file = 'training.pt'
        processed_folder = os.path.join(self.root, 'MNIST', 'processed')
        train_data, train_labels = torch.load(
            os.path.join(self.root, processed_folder, training_file))
        # 随机化数据
        if shuffle:
            random.seed(50)
            random.shuffle(train_data)
            random.seed(50)
            random.shuffle(train_labels)

        train_data_size = len(train_labels)
        average_train_size = int(train_data_size / num_worker)
        loaders = []
        for index in range(num_worker):
            item_data = train_data[index * average_train_size:(index + 1) * average_train_size]
            item_lable = train_labels[index * average_train_size:(index + 1) * average_train_size]
            train_dataset = CustomDataset(item_data, item_lable, transform=mnist_transforms)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
            loaders.append(train_loader)

        return loaders

    def build_dataset_with_power(self, computing_power, shuffle=False):
        # [0.1,0.2,0.3,0.4]
        real_computing_power = [1 / item_power for item_power in computing_power]
        sum_power = sum(real_computing_power)
        power = [item_power / sum_power for item_power in real_computing_power]

        print('build data')
        mnist_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        training_file = 'training.pt'
        processed_folder = os.path.join(self.root, 'MNIST', 'processed')
        train_data, train_labels = torch.load(
            os.path.join(self.root, processed_folder, training_file))
        # 随机化数据
        if shuffle:
            random.seed(50)
            random.shuffle(train_data)
            random.seed(50)
            random.shuffle(train_labels)

        train_data_size = len(train_labels)
        length = [int(train_data_size * item_power) for item_power in power]
        loaders = []
        num_worker = len(computing_power)
        cur_start_index = 0
        for index in range(num_worker):
            item_data = train_data[cur_start_index:cur_start_index + length[index]]
            item_lable = train_labels[cur_start_index:cur_start_index + length[index]]
            cur_start_index = cur_start_index + length[index]
            train_dataset = CustomDataset(item_data, item_lable, transform=mnist_transforms)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
            loaders.append(train_loader)

        return loaders

    def get_test_loader(self):
        print('build test data')
        mnist_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        test_file = 'test.pt'
        processed_folder = os.path.join(self.root, 'MNIST', 'processed')
        test_data, test_labels = torch.load(os.path.join(self.root, processed_folder, test_file))

        test_list = list(zip(test_data, test_labels))
        random.shuffle(test_list)
        test_data, test_labels = zip(*test_list)
        test_dataset = CustomDataset(test_data, test_labels, transform=mnist_transforms)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)

        return test_loader
