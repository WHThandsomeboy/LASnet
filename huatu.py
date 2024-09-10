# import matplotlib.pyplot as plt
# import numpy as np
# import pyedflib
# import stft
# import torch
# from scipy import signal
#
# from SPECTOGRAM import butter_bandstop_filter, butter_highpass_filter
#
# # 其实每一个edf都有18+个通道
# f = pyedflib.EdfReader('/d/CHB-MIT/chb-mit/' + 'chb01' + '/' + 'chb01_03.edf')
# c = ''
# time = ''
# # 这是循环读前18个通道
# channel = 18
# a = f.readSignal(channel)
# b = a[:256 * 5]  # 239*224 -> 224*224
# fs = 256
# lowcut = 117
# highcut = 123
# # 滤波
# y = butter_bandstop_filter(b, lowcut, highcut, fs, order=6)
# lowcut = 57
# highcut = 63
# y = butter_bandstop_filter(y, lowcut, highcut, fs, order=6)
# cutoff = 1
# y = butter_highpass_filter(y, cutoff, fs, order=6)
#
# Pxx = signal.spectrogram(y, nfft=256, fs=256, return_onesided=True, noverlap=128)[2]
# # stft_data = stft.spectrogram(y, framelength=256, centered=False)
# # 去除干扰信号
# Pxx = np.delete(Pxx, np.s_[117:123 + 1], axis=0)
# Pxx = np.delete(Pxx, np.s_[57:63 + 1], axis=0)
# Pxx = np.delete(Pxx, 0, axis=0)
# # Pxx = np.delete(Pxx, np.s_[0:3], axis=0)
# # Pxx = np.delete(Pxx, np.s_[0:7], axis=1)
#
# result = (10 * np.log10(np.transpose(Pxx)) - (10 * np.log10(np.transpose(Pxx))).min()) / (
#         10 * np.log10(np.transpose(Pxx))).ptp()
#
# print(Pxx.shape)
# freqs = np.arange(result.shape[1])
# bins = np.arange(result.shape[0])
# plt.figure(figsize=(10, 5))  # 设置图形大小为宽10英寸，高5英寸
# plt.pcolormesh(freqs, bins, result, cmap=plt.cm.jet)
# # plt.axis('off')  # 关闭坐标轴显示
# plt.colorbar()
# plt.ylabel('Time(s)')
# plt.xlabel('Frequency (Hz)')
# # plt.savefig('./chb-mit.svg', format='svg', dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()

# ---------------------------------------------------------------------------------
# """
# 画eeg图
# """
# import pyedflib
#
# # 读取EDF文件
# file_path = '/d/CHB-MIT/chb-mit/' + 'chb01' + '/' + 'chb01_03.edf'  # 请替换为您的EDF文件路径
# f = pyedflib.EdfReader(file_path)
# a = f.readSignal(1)
# import matplotlib.pyplot as plt
# # Seizure Start Time: 2996 seconds
# # Seizure End Time: 3036 seconds
# start = 256 * 2850
# end = 256 * 3200
# # 假设这是你的浮点数列表
# data = a[start:end]
#
# # 创建 x 轴数据，假设 x 轴是数据的索引
# x = range(len(data))
# plt.figure(figsize=(20, 3))  # 设置宽度为 10，高度为 5
# # 使用 plt.plot 函数绘制折线图
# plt.plot(x, data)  # marker='o' 是为了在数据点上添加圆圈标记
# plt.axis('off')  # 关闭坐标轴显示
# # plt.savefig('./line_plot.svg', format='svg', dpi=300, bbox_inches='tight')
# plt.show()  # 显示绘制的图像

"""
绘制多通道EEG
"""
# import matplotlib.pyplot as plt
# import numpy as np
# import pyedflib
#
# # 读取EDF文件
# file_path = '/d/CHB-MIT/chb-mit/' + 'chb01' + '/' + 'chb01_03.edf'  # 请替换为您的EDF文件路径
# f = pyedflib.EdfReader(file_path)
#
# # 选择要读取的信号通道，这里假设选择前三个通道
# num_channels = 5
# signals = [f.readSignal(i) for i in range(num_channels)]
#
# # 时间范围设置
# start = 256 * 2850
# end = 256 * 2960
# data = [signal[start:end] for signal in signals]
#
# # 创建 x 轴数据，假设 x 轴是数据的索引
# x = np.arange(len(data[0]))  # 使用第一个通道的长度作为 x 轴数据
#
# plt.figure(figsize=(30, 2 * num_channels))  # 设置宽度为 20，高度为 6*num_channels
#
# # 使用 plt.plot 函数绘制折线图
# for i in range(num_channels):
#     plt.subplot(num_channels, 1, i + 1)
#     plt.plot(x, data[i], color="#303030", label=f'Channel {i + 1}')  # marker='o' 是为了在数据点上添加圆圈标记
#     plt.axis('off')  # 关闭坐标轴显示
#     plt.grid(True)  # 添加网格
#     # plt.legend()  # 显示图例
# plt.savefig('/e/wht_project/muti_eeg.svg', format='svg', dpi=300, bbox_inches='tight')
# plt.show()  # 显示绘制的图像

"""
画选定K值模型精度图
"""
# import re
#
# import matplotlib.pyplot as plt
#
#
# # Function to extract data from the log file
# def extract_data(file_path):
#     with open(file_path, 'r') as file:
#         log_content = file.read()
#
#     # Extracting relevant information using regular expressions
#     matches = re.findall(r'Accuracy on test: ([\d.]+).*?Patient: (\w+)', log_content, re.DOTALL)
#
#     # Formatting the data
#     data = [{"accuracy": float(match[0]), "patient": match[1]} for match in matches]
#
#     return data
#
#
# # dataset = "AES"
# # dataset1 = "Kaggle"
# dataset = "CHB-MIT"
#
# # Specify the path to your log file
# K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18]
# data_sets = {}
# for k in K:
#     log_file_path = f'/e/wht_project/eeg_data/test/K={k}_channel_models3/{dataset}.log'
#
#     data = extract_data(log_file_path)
#     data_sets.update({f"K={k}": data})
#
# # Plotting the data for each set with different colors
# plt.figure(figsize=(10, 6))
#
# for label, data in data_sets.items():
#     patients = [entry["patient"] for entry in data]
#     accuracies = [entry["accuracy"] for entry in data]
#     plt.plot(patients, accuracies, marker='o', linestyle='-', label=label)
#
# plt.title('Accuracy for Each Object (Different K values)')
# plt.xlabel(f'{dataset}')
# plt.ylabel('Accuracy')
# plt.legend()
# # plt.savefig(f'/e/wht_project/eeg_data/{dataset}_select_K.svg', format='svg', dpi=300, bbox_inches='tight')
# plt.show()

"""
画柱状图
"""
# 三个东西
# acc flops 柱子 acc/flops
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# # 从日志文件中读取数据
# def read_log_file(file_path):
#     K_values = []
#     accuracy_values = []
#     flops_values = []
#     with open(file_path, 'r') as file:
#         current_K = None
#         current_accuracy = None
#         current_flops = None
#         for line in file:
#             if line.startswith('K='):
#                 if current_K is not None and current_accuracy is not None and current_flops is not None:
#                     K_values.append(current_K)
#                     accuracy_values.append(current_accuracy)
#                     flops_values.append(current_flops)
#                 current_K = int(line.split('=')[1])
#             elif line.startswith('acc:'):
#                 current_accuracy = float(line.split(':')[1])
#             elif line.startswith('Total Flops:'):
#                 current_flops = float(line.split(':')[1][:-7])  # 去掉末尾的 'MFlops' 并转换为浮点数
#         # 添加最后一组数据
#         if current_K is not None and current_accuracy is not None and current_flops is not None:
#             K_values.append(current_K)
#             accuracy_values.append(current_accuracy)
#             flops_values.append(current_flops)
#     return K_values, accuracy_values, flops_values
#
#
# # 计算准确率与总的浮点运算次数的比值
# def calculate_accuracy_flops_ratio(accuracy_values, flops_values):
#     ratio_values = [accuracy / flops for accuracy, flops in zip(accuracy_values, flops_values)]
#     return ratio_values
#
#
# # 日志文件路径
# log_file_path = '/e/wht_project/eeg_data/test2/extracted_data.log'
#
# # 从日志文件中读取数据
# K_values, accuracy_values, flops_values = read_log_file(log_file_path)
#
# # 计算准确率与总的浮点运算次数的比值
# ratio_values = calculate_accuracy_flops_ratio(accuracy_values, flops_values)
#
# # 创建图表
# fig, ax1 = plt.subplots(figsize=(10, 6))
#
# # 绘制准确率和Flops的柱状图
# ax1.bar(np.array(K_values) - 0.2, accuracy_values, width=0.4, color='#6493C6', label='Accuracy (%)')
# ax1.bar(np.array(K_values) + 0.2, flops_values, width=0.4, color='#FA7F6F', label='Total Flops (MFlops)')
# ax1.set_xlabel('K Values')
# ax1.set_ylabel('Values')
# # ax1.set_title('Accuracy and Total Flops vs K Values')
# ax1.set_title('AES')
# ax1.set_xticks(K_values)  # 设置x轴刻度为K值
# ax1.legend(loc='upper right')
# # 设置第一个轴（ax1）的纵坐标范围和刻度
# ax1.set_ylim(0, 100)
# ax1.set_yticks(np.arange(0, 101, 20))
# # 创建第二个y轴用于绘制Accuracy/Flops的折线图
# ax2 = ax1.twinx()
# ax2.plot(K_values, ratio_values, color='#E3C478', marker='o', label='Accuracy/Flops')
# ax2.set_ylabel('Accuracy/Flops')
# ax2.legend(loc='upper left')
# # 将第二个轴（ax2）移动到右上角
# plt.savefig("/e/wht_project/eeg_data/test2/acc_flops_ratio.svg")
#
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# 通道名字
channels = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ"
]

# 给定的数据
data = [
    [3, 14, 8, 15],
    [17, 0, 8, 10],
    [8, 13, 16, 11],
    [0, 1, 17, 2],
    [10, 12, 8, 2],
    [9, 4, 15, 2],
    [17, 9, 15, 7],
    [1, 10, 7, 5],
    [15, 17, 13, 8],
    [10, 15, 0, 5],
    [2, 5, 16, 6],
    [6, 5, 4, 15],
    [9, 10, 5, 14],
    [0, 14, 12, 17],
    [0, 3, 5, 15],
    [4, 10, 9, 0],
    [11, 10, 15, 12],
    [14, 7, 3, 4],
    [0, 9, 8, 3],
    [2, 4, 12, 11],
    [16, 15, 10, 1]
]

# 统计每个通道的数据出现的次数
counts = [0] * len(channels)
for sublist in data:
    for num in sublist:
        counts[num] += 1

# 生成柱状图
plt.bar(channels, counts)
plt.xlabel('Channel')
plt.ylabel('Frequency')
plt.title('Frequency of Channels')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("/e/wht_project/Frequency_of_Channels.svg")
plt.show()
