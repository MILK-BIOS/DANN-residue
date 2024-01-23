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
        #self.feature_mean = np.mean(data, axis=0)
        #self.feature_std = np.std(data, axis=0)
        #print(self.feature_mean)
        #print(self.feature_std)
        self.feature_mean = np.array([-1.5640e+01, -3.0492e+01,  3.5560e+01,  6.8423e-01,  2.7796e+04,
          5.2921e-01,  8.3131e+02,  1.1490e+03,  1.6269e+03,  2.3394e+03,
          2.7287e+03,  7.9246e-02, -2.5195e-01, -7.6281e-02,  1.1767e+00])#这是lishu_s12_21图像的mean
        self.feature_std = np.array([6.8390e+01, 1.0825e+02, 1.2776e+02, 6.2501e+02, 3.6862e+07, 2.9474e+02,
         1.7381e+02, 2.3112e+02, 3.3107e+02, 4.5758e+02, 5.4512e+02, 4.1461e-02,
         8.5349e-02, 9.1382e-02, 1.0025e-01])
        #后续再改的时候，为源域和目标域分别加各自图像的mean和std，这一点很重要

    def __call__(self, sample):
        sample = (sample - self.feature_mean) / self.feature_std  # 归一化
        return sample

class CustomTransform_target:
    def __init__(self):
        self.feature_mean = None
        self.feature_std = None

    def fit(self, data):
        #self.feature_mean = np.mean(data, axis=0)
        #self.feature_std = np.std(data, axis=0)
        #print(self.feature_mean)
        #print(self.feature_std)
        self.feature_mean = np.array([-1.6417e+01, -3.1448e+01,  3.6362e+01,  4.5159e-01,  1.2899e+03,
          7.0286e-01,  7.9835e+02,  1.2558e+03,  1.9334e+03,  2.9812e+03,
          2.6952e+03,  1.4174e-01, -1.7032e-01,  4.4627e-02,  1.3331e+00])
        self.feature_std = np.array([1.9450e+01, 3.2024e+01, 3.7120e+01, 3.9507e-01, 2.8748e+03, 2.2344e-01,
         1.6949e+02, 2.2510e+02, 3.3402e+02, 4.7317e+02, 2.0759e+02, 2.9258e-02,
         7.6024e-02, 7.4289e-02, 7.8293e-02])

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
        col = [i for i in self.df.columns if i in ['VVresidue', 'VHresidue', 'span', 'ratio', 'product','rvi','B2','B3','B4','B8','B12','NDTI','NDRI','NDI7','STI']]
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
        #print(f"Original data_row: {data_row}")
        #print(f"Type of data_row: {type(data_row)}")

        # 判断是源域还是目标域
        if self.noise_level_source > 0.0:  # 如果是源域，且有噪声水平
            data_row += np.random.normal(0, self.noise_level_source, len(data_row))

        if self.noise_level_target > 0.0:
            data_row += np.random.normal(0, self.noise_level_target, len(data_row))

        # 转为张量
        sample_tensor = torch.tensor(np.array(data_row))
        #data_row_np = data_row.cpu().numpy() if data_row.is_cuda else data_row.numpy()
        #print(
        #    f"Any NaN or non-numeric values in data_row: {np.isnan(data_row_np).any() or not np.isreal(data_row_np).all()}")

        return sample_tensor, labels

    def __len__(self):
        return len(self.df)


