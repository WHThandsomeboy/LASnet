import os
import sys

from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init

# from Few_spike.fs_coding import replace_relu_with_fs
from dataProcess.MIT_process_k_flod import cross_validation, MITDataset
# from kaggle_data_process import targets
from load_data import patients


# replace_relu_with_fs()


class SEAttention(nn.Module):
    def __init__(self, channel=18, K=1, reduction=3):
        super().__init__()
        self.K = K
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False),
                                nn.Sigmoid())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        select_c = self.K
        topk_values, topk_indices = torch.topk(y, k=select_c, dim=1)

        sort_indices = torch.argsort(topk_indices, dim=1)
        batch_indices = torch.arange(y.shape[0]).view(-1, 1)

        # 使用排序索引来重新排序topk_indices和topk_values
        sorted_topk_indices = torch.gather(topk_indices, 1, sort_indices)
        sorted_topk_values = torch.gather(topk_values, 1, sort_indices)
        sorted_topk_indices = sorted_topk_indices.reshape(y.shape[0], select_c)
        # 选择原始数据中对应的通道数据
        new_output = x[batch_indices, sorted_topk_indices]
        # new_output = torch.multiply(sorted_topk_values, new_output)
        return new_output * sorted_topk_values.expand_as(new_output)


class Attention(nn.Module):
    def __init__(self, in_channels, K):
        super(Attention, self).__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=3,
                              padding=1,
                              bias=False)
        self.K = K
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.avg_pool(x)
        max_output = self.max_pool(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        max_output = max_output.squeeze(-1).transpose(-1, -2)
        max_output = self.con1(max_output).transpose(-1, -2).unsqueeze(-1)
        max_output = output + max_output
        max_output = self.act1(max_output)
        # output = torch.multiply(x, max_output)
        # return output

        # select_c = output.shape[1] // self.K
        select_c = self.K
        topk_values, topk_indices = torch.topk(max_output, k=select_c, dim=1)
        # print(topk_indices[1].reshape(4).tolist())

        sort_indices = torch.argsort(topk_indices, dim=1)
        batch_indices = torch.arange(max_output.shape[0]).view(-1, 1)

        # 使用排序索引来重新排序topk_indices和topk_values
        sorted_topk_indices = torch.gather(topk_indices, 1, sort_indices)
        sorted_topk_values = torch.gather(topk_values, 1, sort_indices)
        sorted_topk_indices = sorted_topk_indices.reshape(max_output.shape[0], select_c)
        # 选择原始数据中对应的通道数据
        new_output = x[batch_indices, sorted_topk_indices]
        new_output = torch.multiply(sorted_topk_values, new_output)
        return new_output
        # topk_indices = topk_indices.reshape(max_output.shape[0], select_c)
        # batch_indices = torch.arange(max_output.shape[0]).view(-1, 1)
        # # 选择原始数据中对应的通道数据
        # new_output = x[batch_indices, topk_indices]
        # new_output = torch.multiply(topk_values, new_output)
        # # 对张量进行排序
        # sorted_tensor, _ = torch.sort(new_output, dim=1)
        # return new_output


class LightweightCNN(nn.Module):
    def __init__(self, in_channels, K, num_classes=2):
        super(LightweightCNN, self).__init__()
        # 定义深度可分离卷积
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.K = K
        self.CA = Attention(in_channels, self.K)
        # self.CA2 = SEAttention(in_channels, self.K)
        # 定义卷积层
        self.conv1 = nn.Conv2d(self.K, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义全连接层
        if in_channels == 18:  # MIT
            self.fc1 = nn.Linear(64 * 2 * 28, 128)
        elif in_channels == 16:  # Kaggle
            self.fc1 = nn.Linear(64 * 2 * 25, 128)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.CA(x)
        # x = self.CA2(x)
        # x = x[:, :self.K]
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    Dataset = "CHB-MIT"

    # 检查是否有可用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = patients
    seed_all(50)
    for indexPat in range(0, len(dataset)):
        windows = 5
        patients_datas, patients_labels, kf = cross_validation(indexPat)
        for k_fold, (train_index, val_index) in enumerate(kf.split(patients_datas)):
            train_data, val_data = patients_datas[train_index], patients_datas[val_index]
            train_labels, val_labels = [patients_labels[i] for i in train_index], [patients_labels[i] for i in
                                                                                   val_index]

            train_dataset = MITDataset(torch.Tensor(train_data), torch.LongTensor(train_labels))
            val_dataset = MITDataset(torch.Tensor(val_data), torch.LongTensor(val_labels))

            train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=False, num_workers=8)
            val_loader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=False, num_workers=8)

            print(f"Fold {k_fold + 1}")
            print("训练集大小:", len(train_dataset))
            print("验证集大小:", len(val_dataset))

            import logging

            logging.basicConfig(level=logging.INFO)  # 将日志记录到指定路径的文件
            K = [1, 4, 8, 15, 18]
            for k in K:
                print("K=", k)
                path = f"/e/wht_project/eeg_data/k_fold/K={k}_channel_models"
                os.makedirs(path, exist_ok=True)

                # 设置每个病人每一折的日志文件路径
                log_file_path = f'{path}/Patient_{indexPat + 1}_Fold_{k_fold}_K_{k}.log'

                # 配置日志记录
                logging.basicConfig(level=logging.INFO)
                file_handler = logging.FileHandler(log_file_path, mode='a')
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                logging.getLogger().addHandler(file_handler)

                # 创建模型
                model = LightweightCNN(in_channels=18, K=k)
                model.to(device)

                # 定义损失函数和优化器
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                # 训练循环
                num_epochs = 50  # 设置训练的轮数
                best_acc = 0
                best_val_acc = 0
                start = time.time()

                for epoch in range(num_epochs):
                    model.train()  # 进入训练模式
                    running_loss = 0.0
                    correct_train = 0
                    total_train = 0

                    for i, data in enumerate(train_loader, 0):
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU

                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total_train += labels.size(0)
                        correct_train += (predicted == labels).sum().item()

                    train_accuracy = 100 * correct_train / total_train
                    if train_accuracy > best_acc:
                        best_acc = train_accuracy

                    print(
                        f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_accuracy}%")
                    if train_accuracy >= 99.98:
                        break

                    # 测试模型
                    model.eval()  # 进入评估模式
                    correct = 0
                    total = 0
                    TP = 0
                    TN = 0
                    FN = 0
                    FP = 0

                    with torch.no_grad():
                        for x, y in val_loader:
                            x, y = x.to(device), y.to(device)
                            outputs = model(x)
                            _, predicted = torch.max(outputs.data, 1)
                            total += y.size(0)
                            correct += (predicted == y).sum().item()
                            c = (predicted == y)
                            for i in range(predicted.shape[0]):
                                if (c[i] == 1).item() == 1:
                                    if y[i] == 1:
                                        TP += 1
                                    elif y[i] == 0:
                                        TN += 1
                                elif (c[i] == 1).item() == 0:
                                    if y[i] == 1:
                                        FN += 1
                                    elif y[i] == 0:
                                        FP += 1

                        val_acc = 100 * correct / total
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            torch.save(model.state_dict(), f"{path}/Patient_{indexPat + 1}_Fold_{k_fold}_K_{k}.pth")

                        print(f'Accuracy on val: {val_acc:.2f}%')

                end = time.time()
                print(f"Training Finished")
                print(f"Training time: {end - start}")
                logging.info(f'Training time: {end - start}')

                # 计算并记录评估指标
                try:
                    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    F1_score = (2 * sensitivity * precision) / (sensitivity + precision) if (
                                                                                                    sensitivity + precision) > 0 else 0
                    logging.info(f'sensitivity/recall: {sensitivity * 100:.2f}%')
                    logging.info(f'specificity: {specificity * 100:.2f}%')
                    logging.info(f'precision: {precision * 100:.2f}%')
                    logging.info(f'F1_score: {F1_score * 100:.2f}%')

                    print("sensitivity/recall:", sensitivity * 100)
                    print("specificity:", specificity * 100)
                    print("precision:", precision * 100)
                    print("F1_score:", F1_score * 100)
                except Exception as e:
                    print(f"Error in calculating metrics: {e}")

                logging.info(f'Patient: {indexPat + 1}')
                logging.info(f'Best validation accuracy: {best_val_acc}')
                logging.info('------------------------------------------------')

                # 关闭文件处理程序，以确保缓冲的日志被写入文件
                file_handler.close()
                logging.getLogger().removeHandler(file_handler)

            del train_loader, val_loader
