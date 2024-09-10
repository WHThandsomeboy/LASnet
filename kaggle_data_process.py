import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.save_load import load_hickle_file

with open('SETTINGS_%s.json' % "Kaggle") as f:
    settings = json.load(f)

# 这是真正的指定处
targets = [
    'Dog_1',
    'Dog_2',
    'Dog_3',
    'Dog_4',
    'Dog_5',
]


class KaggleDataset(Dataset):
    train_len, val_len, test_len, channel, train_dataset, val_dataset, test_dataset, train_label, val_label, test_label, patient_index = None, None, None, None, None, None, None, None, None, None, None

    def __init__(self, dataset: str, patient_index: int):
        """
        训练数据集与测试数据集的Dataset对象
        :param dataset: 区分是获得训练集还是测试集
        :param patient_index: 判断是kaggle数据集中哪个对象
        """
        super(KaggleDataset, self).__init__()
        self.dataset = dataset  # 选择获取测试集还是训练集
        self.patient_index = patient_index

        # 如果已经初始化过，则无需再调用pre_option()
        if (KaggleDataset.train_len is None and KaggleDataset.patient_index is None) or (
                KaggleDataset.patient_index is not None and KaggleDataset.patient_index != self.patient_index):
            KaggleDataset.train_len, KaggleDataset.val_len, KaggleDataset.test_len, KaggleDataset.channel, KaggleDataset.train_dataset, KaggleDataset.val_dataset, KaggleDataset.test_dataset, KaggleDataset.train_label, KaggleDataset.val_label, KaggleDataset.test_label, KaggleDataset.total_interictal, KaggleDataset.patient_index = self.pre_option()

    def __getitem__(self, index):
        if self.dataset == 'train':
            return self.train_dataset[index], self.train_label[index]
        elif self.dataset == 'val':
            return self.val_dataset[index], self.val_label[index]
        elif self.dataset == 'test':
            return self.test_dataset[index], self.test_label[index]

    def __len__(self):
        if self.dataset == 'train':
            return self.train_len
        elif self.dataset == 'val':
            return self.val_len
        elif self.dataset == 'test':
            return self.test_len

    # 数据预处理
    def pre_option(self):
        target = targets[self.patient_index]
        ictal_X, ictal_y = load_hickle_file(os.path.join(settings['cachedir'], f'ictal_{target}'))

        X_train_ictal = np.concatenate(ictal_X[:], axis=0).astype(np.float32)
        y_train_ictal = np.concatenate(ictal_y[:], axis=0).astype(np.float32)
        del ictal_X, ictal_y
        interictal_X, interictal_y = load_hickle_file(os.path.join(settings['cachedir'], f'interictal_{target}'))
        if isinstance(interictal_y, list):
            interictal_X = np.concatenate(interictal_X, axis=0).astype(np.float32)  # nd(?,1,256*8,22)
            interictal_y = np.concatenate(interictal_y, axis=0).astype(np.float32)
        # total_interictal = interictal_y.shape[0]
        X_train_interictal = None
        y_train_interictal = None
        down_spl = int(np.floor(interictal_y.shape[0] / y_train_ictal.shape[0]))
        if down_spl > 1:
            X_train_interictal = interictal_X[::down_spl]
            y_train_interictal = interictal_y[::down_spl]
        elif down_spl == 1:
            X_train_interictal = interictal_X[:X_train_ictal.shape[0]]
            y_train_interictal = interictal_y[:X_train_ictal.shape[0]]
        print('balancing y_train_ictal.shape, y_train_interictal.shape: ', X_train_ictal.shape,
              y_train_interictal.shape)
        # total_interictal = y_train_interictal.shape[0]
        del interictal_X, interictal_y
        y_train_ictal[y_train_ictal == 2] = 1
        all_data = np.concatenate((X_train_ictal, X_train_interictal), axis=0).astype(np.float32)
        del X_train_ictal, X_train_interictal
        all_label = np.concatenate((y_train_ictal, y_train_interictal), axis=0).astype(np.float32)
        del y_train_ictal, y_train_interictal
        # 打乱所有数据和标签
        # 生成一个随机排列的索引数组
        random_indices = np.random.permutation(all_data.shape[0])
        # 使用这些随机索引对训练数据集和标签集进行重新排列
        shuffled_data = all_data[random_indices]
        shuffled_label = [all_label[i] for i in random_indices]
        del all_data
        # 划分训练集、验证集和测试集
        train_percent = 25  # 25-100
        val_percent = 10  # 10-25
        start_ = int(len(shuffled_data) / 100 * train_percent)
        start_val = int(len(shuffled_data) / 100 * val_percent)
        train, train_label = shuffled_data[start_:], shuffled_label[start_:]
        val, val_label = shuffled_data[start_val:start_], shuffled_label[start_val:start_]
        test, test_label = shuffled_data[:start_val], shuffled_label[:start_val]
        total_interictal = test_label.count(0)
        train_len, val_len, test_len = train.shape[0], val.shape[0], test.shape[0]
        train, train_label = torch.Tensor(train), torch.LongTensor(train_label)
        val, val_label = torch.Tensor(val), torch.LongTensor(val_label)
        test, test_label = torch.Tensor(test), torch.LongTensor(test_label)
        channel = train[0].shape[0]
        return train_len, val_len, test_len, channel, train, val, test, train_label, val_label, test_label, total_interictal, self.patient_index
