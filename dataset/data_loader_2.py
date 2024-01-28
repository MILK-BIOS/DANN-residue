import numpy as np
import torch
import torch.utils.data as data
import os
import pandas as pd
import torchvision.transforms as transforms


class CustomTransform_source:
    def __init__(self):
        self.feature_mean = None
        self.feature_std = None

    def fit(self, data):
        # self.feature_mean = np.mean(data, axis=0)
        # self.feature_std = np.std(data, axis=0)
        # print(self.feature_mean)
        # print(self.feature_std)
        self.feature_mean = np.array(
            [-15.593433380126953, -30.415849685668945, 35.33893585205078, 0.30355292558670044, 1372.9437255859375,
             0.5820470452308655, 831.2803344726562, 1148.9268798828125, 1626.899658203125, 2339.423095703125,
             2728.666015625, 0.07924484461545944, -0.2519519627094269, -0.07628648728132248,
             1.1767325401306152])  # 这是lishu_s12_21图像的mean
        self.feature_std = np.array(
            [33.18778991699219, 55.5806999206543, 35.336002349853516, 49.58562469482422, 83712.9453125,
             32.980770111083984, 173.60861206054688, 230.97706604003906, 330.97149658203125, 457.4918518066406,
             544.689697265625, 0.04145393148064613, 0.0853317379951477, 0.09136494994163513, 0.1002340242266655])
        # 后续再改的时候，为源域和目标域分别加各自图像的mean和std，这一点很重要

    def __call__(self, sample):
        sample = (sample - self.feature_mean) / self.feature_std  # 归一化
        return sample


class CustomTransform_target:
    def __init__(self):
        self.feature_mean = None
        self.feature_std = None

    def fit(self, data):
        # self.feature_mean = np.mean(data, axis=0)
        # self.feature_std = np.std(data, axis=0)
        # print(self.feature_mean)
        # print(self.feature_std)
        self.feature_mean = np.array(
            [-14.884668350219727, -28.888315200805664, 33.29230499267578, 0.4585520923137665, 927.4970703125, 0.6962623000144958, 797.4642944335938, 1255.49365234375, 1934.1185302734375, 2983.580810546875, 2697.75048828125, 0.1424081027507782, -0.16956187784671783, 0.04584725201129913, 1.3344955444335938])
        self.feature_std = np.array(
            [15.705412864685059, 25.763347625732422, 29.83693504333496, 0.3049178719520569, 1869.6712646484375, 0.1617993265390396, 152.79661560058594, 203.365234375, 303.1920166015625, 430.80657958984375, 179.0396728515625, 0.026517020538449287, 0.06928186118602753, 0.0676175057888031, 0.07123342156410217])

    def __call__(self, sample):
        sample = (sample - self.feature_mean) / self.feature_std  # 归一化
        return sample


class load_data(data.Dataset):
    def __init__(self, data_root, data_set, transform=True, noise_level_source=0, noise_level_target=0, domain_flag=0):
        self.root = data_root
        self.transform = transform
        self.noise_level_source = noise_level_source
        self.noise_level_target = noise_level_target
        self.domain_flag = domain_flag
        self.df = pd.read_csv(self.root)
        col = [i for i in self.df.columns if
               i in ['VVresidue', 'VHresidue', 'span', 'ratio', 'product', 'rvi', 'B2', 'B3', 'B4', 'B8', 'B12', 'NDTI',
                     'NDRI', 'NDI7', 'STI']]
        self.X = self.df[col]

        label_mapping = {'TC': 0, 'RC': 1, 'NC': 2}
        if 'residue' in self.df:
            self.labels = self.df['residue'].map(label_mapping)
        else:
            self.labels = pd.Series(np.zeros(len(self.df)))

        self.custom_transform_source = CustomTransform_source()
        self.custom_transform_source.fit(self.X.values)
        self.custom_transform_target = CustomTransform_target()
        self.custom_transform_target.fit(self.X.values)

    def __getitem__(self, item):
        data_row = self.X.loc[item].values.astype(np.float64)  # 将data_row转换为 float64 的NumPy数组
        labels = int(self.labels.loc[item])
        if self.domain_flag == 0:
            data_row = self.custom_transform_source(data_row)
        else:
            data_row = self.custom_transform_target(data_row)
        # print(f"Original data_row: {data_row}")
        # print(f"Type of data_row: {type(data_row)}")

        # 判断是源域还是目标域
        if self.noise_level_source > 0.0:  # 如果是源域，且有噪声水平
            data_row += np.random.normal(0, self.noise_level_source, len(data_row))

        if self.noise_level_target > 0.0:
            data_row += np.random.normal(0, self.noise_level_target, len(data_row))

        # 转为张量
        sample_tensor = torch.tensor(np.array(data_row))
        # data_row_np = data_row.cpu().numpy() if data_row.is_cuda else data_row.numpy()
        # print(
        #    f"Any NaN or non-numeric values in data_row: {np.isnan(data_row_np).any() or not np.isreal(data_row_np).all()}")

        return sample_tensor, labels

    def __len__(self):
        return len(self.df)
