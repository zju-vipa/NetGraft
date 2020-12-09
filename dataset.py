import os
import pickle
import sys
import random

import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class CIFAR10Few(Dataset):
    def __init__(self, root, num_per_class=50, transform=None):

        self.root = root

        self.transform = transform

        filename = 'cifar10-random-{}-per-class.pkl'.format(num_per_class)

        file_path = os.path.join(self.root, filename)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f)

        self.data = entry

        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        img = self.data[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)


def load_cifar10(root):
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    base_folder = 'cifar-10-batches-py'

    data = []
    targets = []
    for file_name, _ in train_list:
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
            if 'labels' in entry:
                targets.extend(entry['labels'])
            else:
                targets.extend(entry['fine_labels'])
    data = np.vstack(data)
    return data, targets


def categorize(data, targets, num_class):
    data_by_class = [[] for i in range(num_class)]
    for i, target in enumerate(targets):
        data_by_class[target].append(data[i])

    for i in range(num_class):
        data_by_class[i] = np.vstack(data_by_class[i])

    return data_by_class


def extract_dataset_from_cifar10(num_per_class):
    data, targets = load_cifar10('data')

    num_class = 10
    data_by_class = categorize(data, targets, num_class)

    random.seed(6)
    data_select = []
    for i in range(num_class):
        idxs = list(range(5000))
        random.shuffle(idxs)
        idx_select = idxs[:num_per_class]
        data_select.append(data_by_class[i][idx_select])

    data_select = np.vstack(data_select).reshape(-1, 3, 32, 32)
    
    with open('data/cifar10-random-{}-per-class.pkl'.format(num_per_class), \
        'wb') as f:
        pickle.dump(data_select, f)


class CIFAR100Few(Dataset):
    def __init__(self, root, num_per_class=10, transform=None):
        self.root = root

        self.transform = transform

        filename = 'cifar100-random-{}-per-class.pkl'.format(num_per_class)

        file_path = os.path.join(self.root, filename)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f)

        self.data = entry

        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        img = self.data[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)


def load_cifar100(root):
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    base_folder = 'cifar-100-python'

    data = []
    targets = []
    for file_name, _ in train_list:
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            data.append(entry['data'])
            if 'fine_labels' in entry:
                targets.extend(entry['fine_labels'])
            else:
                targets.extend(entry['coarse_labels'])
    data = np.vstack(data)
    return data, targets


def extract_dataset_from_cifar100(num_per_class):
    data, targets = load_cifar100('data')

    num_class = 100
    data_by_class = categorize(data, targets, num_class)

    random.seed(10)
    data_select = []
    for i in range(num_class):
        idxs = list(range(500))
        random.shuffle(idxs)
        idx_select = idxs[:num_per_class]
        data_select.append(data_by_class[i][idx_select])

    data_select = np.vstack(data_select).reshape(-1, 3, 32, 32)

    with open('data/cifar100-random-{}-per-class.pkl'.format(num_per_class), \
              'wb') as f:
        pickle.dump(data_select, f)


if __name__ == '__main__':
    num_samples = list(range(1, 11)) + [20, 50]
    for i in num_samples:
        extract_dataset_from_cifar10(i)


