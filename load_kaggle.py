import json
import os

import numpy as np
import pandas as pd
import scipy.io
import stft
from scipy.signal import resample

from utils.group_seizure_Kaggle import group_seizure
from utils.save_load import save_hickle_file, load_hickle_file


def load_signals_Kaggle(data_dir, target, data_type):
    print('load_signals_Kaggle for Patient', target)

    dir = os.path.join(data_dir, target)
    done = False
    i = 0
    while not done:
        i += 1
        if target == 'Dog_4' and data_type == 'preictal':
            if 18 < i <= 26 or 38 < i <= 43:
                continue
        if i < 10:
            tar = f'000{i}'
        elif i < 100:
            tar = f'00{i}'
        elif i < 1000:
            tar = f'0{i}'
        else:
            tar = f'{i}'

        filename = f"{dir}/{target}_{data_type}_segment_{tar}.mat"
        if os.path.exists(filename):
            data = scipy.io.loadmat(filename)  # {dict:4}
            # discard preictal segments from 66 to 35 min prior to seizure
            if data_type == 'preictal':
                for skey in data.keys():
                    if "_segment_" in skey.lower():
                        mykey = skey
                sequence = data[mykey][0][0][4][0][0]
                if sequence <= 3:
                    print('Skipping %s....' % filename)
                    continue
            yield data
        else:
            if i == 1:
                raise Exception("file %s not found" % filename)
            done = True


class PrepData:
    def __init__(self, target, type, settings, window, SOP):
        self.target = target
        self.settings = settings
        self.type = type
        self.window = window
        self.SOP = SOP

    def read_raw_signal(self):
        if self.settings['dataset'] == 'Kaggle':
            if self.type == 'ictal':
                data_type = 'preictal'
            else:
                data_type = self.type
            return load_signals_Kaggle(self.settings['datadir'], self.target, data_type)

        return None

    def preprocess_Kaggle(self, data_):
        ictal = self.type == 'ictal'
        interictal = self.type == 'interictal'
        if 'Dog_' in self.target:
            targetFrequency = 200  # re-sample to target frequency
            DataSampleSize = targetFrequency
            numts = self.window
        else:
            targetFrequency = 1000
            DataSampleSize = targetFrequency
            numts = self.window
        sampleSizeinSecond = 600  # 10min X 60s

        df_sampling = pd.read_csv(
            'sampling_%s.csv' % self.settings['dataset'],
            header=0, index_col=None)
        trg = self.target
        ictal_ovl_pt = df_sampling[df_sampling.Subject == trg].ictal_ovl.values[0]
        ictal_ovl_len = int(targetFrequency * ictal_ovl_pt * numts)

        def process_raw_data(mat_data):
            print('Loading data')
            X = []
            y = []
            sequences = []

            for segment in mat_data:
                for skey in segment.keys():
                    if "_segment_" in skey.lower():
                        mykey = skey
                if ictal:
                    y_value = 1
                    sequence = segment[mykey][0][0][4][0][0]
                else:
                    y_value = 0

                data = segment[mykey][0][0][0]  # (16,239766)
                sampleFrequency = segment[mykey][0][0][2][0][0]  # Dog：400HZ  patient:5000Hz

                if sampleFrequency > targetFrequency:  # resample to target frequency
                    # float64-->float32 save cpu memory
                    data = resample(data, targetFrequency * sampleSizeinSecond, axis=-1).astype(np.float32)
                data = data.transpose()  # (16,120000)-->(120000,16)
                data = np.pad(data, (0, 16 - data.shape[-1])) if data.shape[-1] < 16 else data

                total_sample = int(data.shape[0] / DataSampleSize / numts) + 1  # 一共采样的次数
                window_len = int(DataSampleSize * numts)

                for i in range(total_sample):

                    if (i + 1) * window_len <= data.shape[0]:
                        s = data[i * window_len:(i + 1) * window_len, :]  # nd(200*8, 16)
                        stft_data = stft.spectrogram(s, framelength=DataSampleSize, centered=False)
                        stft_data = stft_data[1:, :, :]
                        stft_data = np.log10(stft_data)
                        indices = np.where(stft_data <= 0)
                        stft_data[indices] = 0
                        stft_data = np.transpose(stft_data, (2, 1, 0))
                        stft_data = np.abs(stft_data) + 1e-6
                        stft_data = stft_data.reshape(1, stft_data.shape[0], stft_data.shape[1], stft_data.shape[2])
                        X.append(stft_data)
                        y.append(y_value)
                        if ictal:
                            sequences.append(sequence)

                if ictal:
                    # overlapped window
                    i = 1
                    while (window_len + (i + 1) * ictal_ovl_len) <= data.shape[0]:
                        s_ = data[i * ictal_ovl_len:i * ictal_ovl_len + window_len, :]
                        stft_data = stft.spectrogram(s_, framelength=DataSampleSize, centered=False)

                        stft_data = stft_data[1:, :, :]
                        stft_data = np.log10(stft_data)
                        indices = np.where(stft_data <= 0)
                        stft_data[indices] = 0
                        stft_data = np.transpose(stft_data, (2, 1, 0))
                        stft_data = np.abs(stft_data) + 1e-6

                        stft_data = stft_data.reshape(1, stft_data.shape[0], stft_data.shape[1], stft_data.shape[2])

                        X.append(stft_data)
                        y.append(2)
                        sequences.append(sequence)
                        i += 1
            if ictal:
                assert len(X) == len(y)
                assert len(X) == len(sequences)
                X, y = group_seizure(X, y, sequences)
                return X, y
            elif interictal:
                X = np.concatenate(X)
                y = np.array(y)
                return X, y
            else:
                X = np.concatenate(X)
                return X, None

        data = process_raw_data(data_)
        return data

    def apply(self):
        filename = '%s_%s' % (self.type, self.target)  # ictal_Dog_2
        cache = load_hickle_file(
            os.path.join(self.settings['cachedir'], filename))
        if cache is not None:
            return cache

        data = self.read_raw_signal()
        if self.settings['dataset'] == 'Kaggle':
            X, y = self.preprocess_Kaggle(data)
        save_hickle_file(
            os.path.join(self.settings['cachedir'], filename),
            [X, y])
        return X, y


if __name__ == '__main__':
    targets = [
        # 'Dog_1',
        # 'Dog_2',
        'Dog_3',
        'Dog_4',
        'Dog_5',
    ]
    with open('SETTINGS_%s.json' % "Kaggle") as f:
        settings = json.load(f)
    for target in targets:
        _, ictal_y = PrepData(target, type='ictal', settings=settings, window=5, SOP=30).apply()
        # inter_x, inter_y = PrepData(target, type='interictal', settings=settings, window=5,SOP=30).apply()
        # print(f"{target}已完成")
        print(len(ictal_y))
