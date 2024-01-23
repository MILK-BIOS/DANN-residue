import numpy as np
import torch
import torch.utils.data as data
import os
import pandas as pd
import torchvision.transforms as transforms


class load_data(data.Dataset):
    def __init__(self, data_root, data_set, transform=True):
        self.root = data_root
        self.transform = transform
        self.df = pd.read_csv(self.root)
        col = [i for i in self.df.columns if i in ['VVresidue', 'VHresidue', 'span', 'ratio', 'product','rvi','B2','B3','B4','B8','B12','NDTI','NDRI','NDI7','STI']]
        self.X = self.df[col]
        label_mapping = {'TC': 0, 'RC': 1, 'NC': 2}
        if 'residue' in self.df:
            self.labels = self.df['residue'].map(label_mapping)
        else:
            self.labels = pd.Series(np.zeros(len(self.df)))
        self.custom_transform = CustomTransform()
        self.custom_transform.fit(self.X.values)

    def __getitem__(self, item):
        if self.transform:
            transform = transforms.Compose([self.custom_transform])
            data_row = transform(self.X.loc[item])
            labels = int(self.labels.loc[item])

        return data_row, labels

    def __len__(self):
        return len(self.df)


class CustomTransform:
    def __init__(self):
        self.feature_mean = None
        self.feature_std = None

    def fit(self, data):
        self.feature_mean = np.mean(data, axis=0)
        self.feature_std = np.std(data, axis=0)

    def __call__(self, sample):
        sample = (sample - self.feature_mean) / self.feature_std  # 归一化
        sample_tensor = torch.tensor(sample, dtype=torch.float32)  # 转为张量
        return sample_tensor
