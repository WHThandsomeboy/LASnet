import logging
import time

import torch
from torch.utils.data import DataLoader

from MIT_data_process import MITDataset
from kaggle_data_process import targets, KaggleDataset
from load_data import loadSpectogramData, patients
from wht_train import LightweightCNN, seed_all


def get_interitcal(indexPat):
    interictalSpectograms, _, nSeizure = loadSpectogramData(indexPat)
    interictalSpectograms = [item for sublist in interictalSpectograms for item in sublist]
    total_interictal = len(interictalSpectograms) * 100
    windows = 5
    total_inter_time = (total_interictal * windows) / (60 * 60)
    return total_inter_time, nSeizure


# Dataset = "Kaggle"  # CHBMIT or Kaggle
Dataset = "CHB-MIT"

if Dataset == "CHB-MIT":
    dataset = patients
    seed_all(50)
else:
    dataset = targets
    seed_all(50)
# # 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
all_sensitivity = 0
all_specificity = 0
all_precision = 0
all_F1_socre = 0
all_FPR = 0
all_acc = 0
all_auc = 0
for indexPat in range(0, len(dataset)):

    test_dataset = None
    is_Patient = False
    windows = 5
    if Dataset == "CHB-MIT":
        test_dataset = MITDataset('test', indexPat)
        print("测试集大小:", test_dataset.test_len)
        total_inter_time, _ = get_interitcal(indexPat)
        print(f'Total inter time:{total_inter_time:.2f} (/h)')
    elif Dataset == "Kaggle":
        test_dataset = KaggleDataset('test', indexPat)
        print("测试集大小:", test_dataset.test_len)
        total_inter_time = (test_dataset.total_interictal * windows) / (60 * 60)
        print(f'Total inter time:{total_inter_time:.2f} (/h)')
        if "Patient" in dataset[indexPat]:
            is_Patient = True

    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

    # 配置日志记录器
    logging.basicConfig(level=logging.INFO)  # 将日志记录到指定路径的文件
    K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    for k in K:
        # path = f"/e/wht_project/eeg_data/no_attention_models2"
        path = f"/e/wht_project/eeg_data/test_final/K={k}_channel_models"
        log_file_path = f'{path}/{Dataset}_test.log'
        # 配置日志处理程序，将日志写入到文件中，以追加模式打开
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        model = LightweightCNN(in_channels=test_dataset.channel, K=k)
        state_dict = torch.load(f"{path}/{Dataset}_{dataset[indexPat]}.pth", map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)
        start = time.time()
        # 测试模型
        model.eval()  # 进入评估模式
        correct = 0
        total = 0
        all_labels = []  # 用于存储所有标签
        all_probabilities = []  # 用于存储所有正类概率
        with torch.no_grad():
            TP = 0
            TN = 0
            FN = 0
            FP = 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                probabilities = torch.nn.functional.softmax(outputs.data)
                # _, predicted = torch.max(probabilities, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                c = (predicted == y)
                # 保存标签和概率
                all_labels.extend(y.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                for i in range(predicted.shape[0]):
                    # 预测正确的情况
                    if (c[i] == 1).item() == 1:
                        if y[i] == 1:
                            TP += 1  # 被模型预测为正类的正样本
                        elif y[i] == 0:
                            TN += 1  # 被模型预测为负类的负样本
                        else:
                            print("TPTNerror")
                    # 预测错误的情况
                    elif (c[i] == 1).item() == 0:
                        if y[i] == 1:
                            FN += 1  # 被模型预测为负类的正样本
                        elif y[i] == 0:
                            FP += 1  # 被模型预测为正类的负样本
                        else:
                            print("FPFNerror")
                    else:
                        print("error")

            print('TP,TN,FP,FN')
            print(TP, TN, FP, FN)
            print(f'Accuracy on test: %.2f %%' % (100 * correct / total))
            logging.info(f'Accuracy on test: {100 * correct / total}')
            all_acc += (100 * correct / total)
            try:
                sensitivity = (TP / (TP + FN))
                specificity = (TN / (TN + FP))
                precision = (TP / (TP + FP))
                FPR = (FP * windows) / total_inter_time / 3600
                # F1_socre = (2 * sensitivity * precision) / (sensitivity + precision)
                print("sensitivity/recall:", sensitivity * 100)
                print("specificity:", specificity * 100)
                print("precision:", precision * 100)
                # print("F1_socre:", F1_socre * 100)
                print("FPR:", FPR)
                logging.info(f'sensitivity/recall: {sensitivity * 100}')
                logging.info(f'specificity: {specificity * 100}')
                logging.info(f'precision: {precision * 100}')
                # logging.info(f'F1_socre: {F1_socre * 100}')
                logging.info(f'FPR: {FPR}')
                logging.info(f'Patient: {dataset[indexPat]}')
            except Exception:
                print("有问题！")
        del model
        logging.info(f'-----------------------------------------------------------------------')
        end = time.time()
        print("test time:", end - start)
        # 关闭文件处理程序，以确保缓冲的日志被写入文件
        file_handler.close()
        logging.getLogger().removeHandler(file_handler)
    del test_dataset, test_loader
