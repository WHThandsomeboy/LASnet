import numpy as np
import torch
from torch.utils.data import Dataset

from load_data import PathSpectogramFolder, patients, loadSpectogramData

OutputPathModels = "./EggModels"


class MITDataset(Dataset):
    train_len, val_len, test_len, channel, train_dataset, val_dataset, test_dataset, train_label, val_label, test_label, patient_index = None, None, None, None, None, None, None, None, None, None, None

    def __init__(self, dataset: str, patient_index: int):
        """
        训练数据集与测试数据集的Dataset对象
        :param dataset: 区分是获得训练集还是测试集
        :param patient: 判断是哪个病人
        """
        super(MITDataset, self).__init__()
        self.dataset = dataset  # 选择获取测试集还是训练集
        self.patient_index = patient_index

        # 如果已经初始化过，则无需再调用pre_option()
        if (MITDataset.train_len is None and MITDataset.patient_index is None) or (
                MITDataset.patient_index is not None and MITDataset.patient_index != self.patient_index):
            MITDataset.train_len, MITDataset.val_len, MITDataset.test_len, MITDataset.channel, MITDataset.train_dataset, MITDataset.val_dataset, MITDataset.test_dataset, MITDataset.train_label, MITDataset.val_label, MITDataset.test_label, MITDataset.patient_index = self.pre_option()

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
        # for indexPat in range(0, len(patients)):
        print('Patient ' + patients[self.patient_index])
        # if not os.path.exists(OutputPathModels + "resultPat" + patients[indexPat] + "/"):
        #     os.makedirs(OutputPathModels + "resultPat" + patients[indexPat] + "/")
        interictalSpectograms, preictalSpectograms, nSeizure = loadSpectogramData(self.patient_index)
        print('Spectograms data loaded')
        filesPath = []
        # 使用列表推导式展平为一维列表
        interictalSpectograms = [item for sublist in interictalSpectograms for item in sublist]
        preictalSpectograms = [item for sublist in preictalSpectograms for item in sublist]
        filesPath.extend(interictalSpectograms)
        filesPath.extend(preictalSpectograms)
        print(filesPath)
        all_label = []
        from concurrent.futures import ThreadPoolExecutor

        def load_array(file_path):
            label = []
            array = np.load(PathSpectogramFolder + file_path).astype(np.float32)
            label.extend([1 if 'P' in file_path else 0] * array.shape[0])
            return array, label

        with ThreadPoolExecutor(max_workers=16) as executor:
            results_all = list(executor.map(load_array, filesPath[0:len(filesPath)]))

        # 一次性读取所有数据
        all_data = np.concatenate([item[0] for item in results_all], axis=0).astype(np.float32)
        for item in results_all:
            all_label.extend(item[1])
        del results_all
        # 打乱所有数据和标签
        # 生成一个随机排列的索引数组
        random_indices = np.random.permutation(all_data.shape[0])
        # 使用这些随机索引对训练数据集和标签集进行重新排列
        shuffled_data = all_data[random_indices]
        shuffled_label = [all_label[i] for i in random_indices]
        del all_data
        # 划分训练集、验证集和测试集
        train_percent = 75
        val_percent = 90
        to_ = int(len(shuffled_data) / 100 * train_percent)
        to_val = int(len(shuffled_data) / 100 * val_percent)
        train, train_label = shuffled_data[:to_], shuffled_label[:to_]
        val, val_label = shuffled_data[to_:to_val], shuffled_label[to_:to_val]
        test, test_label = shuffled_data[to_val:len(shuffled_data)], shuffled_label[to_val:len(shuffled_data)]
        train_len, val_len, test_len = train.shape[0], val.shape[0], test.shape[0]
        train, train_label = torch.Tensor(train), torch.LongTensor(train_label)
        val, val_label = torch.Tensor(val), torch.LongTensor(val_label)
        test, test_label = torch.Tensor(test), torch.LongTensor(test_label)
        channel = train[0].shape[0]
        del shuffled_data
        return train_len, val_len, test_len, channel, train, val, test, train_label, val_label, test_label, self.patient_index
