import os
import torch.backends.cudnn as cudnn
import torch.utils.data
import tifffile as tf
import torch
import numpy as np
from scipy.stats import norm


def remove_outliers_std(data, n_std=3):
    mean = torch.mean(data, dim=0, keepdim=True)
    std = torch.std(data, dim=0, keepdim=True)
    normal_distribution = norm(loc=mean, scale=std)
    lower_bound = normal_distribution.cdf(mean - n_std * std)
    upper_bound = normal_distribution.cdf(mean + n_std * std)

    # 将异常值限制在边界内
    processed_column = np.clip(data, lower_bound, upper_bound)
    return torch.tensor(processed_column, dtype=torch.float32)


def remove_outliers_percentile_columnwise(tensor, lower_percentile=3, upper_percentile=97):
    # 将张量转换为 NumPy 数组
    numpy_tensor = tensor.numpy()

    # 按列分开张量
    columns = torch.split(tensor, 1, dim=1)

    # 对每列进行异常值处理
    processed_columns = [remove_outliers_percentile(column.squeeze().numpy(), lower_percentile, upper_percentile)
                         for column in columns]
    mean = [mean_col(col) for col in processed_columns]
    std = [std_col(col) for col in processed_columns]

    print(mean)
    print(std)
    return mean, std


def remove_outliers_percentile(column, lower_percentile=0, upper_percentile=100):
    lower_bound = np.percentile(column, lower_percentile)
    upper_bound = np.percentile(column, upper_percentile)
    # 将异常值限制在边界内
    processed_column = np.clip(column, lower_bound, upper_bound)
    processed_column = torch.tensor(processed_column)
    # 指定要删除的值
    values_to_remove = [lower_bound, upper_bound]

    # 创建掩码，标记要保留的元素
    mask = torch.ones_like(processed_column, dtype=torch.bool)

    # 针对每个要删除的值，将相应的位置标记为 False
    for value in values_to_remove:
        mask &= (processed_column != value)
    new_tensor = processed_column[mask]

    return new_tensor


def compute_meanstd(data):
    mean = torch.mean(data, dim=0, keepdim=True)  # 计算图像中非 NaN 像素值的均值。通过索引 img[non_nan_mask]，只考虑非 NaN 的像素值。
    print(mean)
    std = torch.std(data, dim=0, keepdim=True)  # 计算图像中非 NaN 像素值的标准差。
    print(std)


def mean_col(col):
    mean = torch.mean(col, dim=0)
    return float(mean)


def std_col(col):
    std = torch.std(col, dim=0)
    return float(std)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_root = os.path.join('', 'models')
    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""
    path = '../dataset/lishu_s12_20.tiff'
    img_tf = tf.imread(path)
    img = torch.tensor(img_tf).view(15, -1).float()
    img = img.T
    print(img.shape)

    non_nan_mask = ~torch.isnan(img)  # 创建了一个布尔掩码（Boolean mask），用于标识图像中非 NaN（Not a Number）的位置。
    # ~ 操作符对这个张量进行按位取反，得到非 NaN 的位置为 True，NaN 的位置为 False。
    nan_mask = torch.isnan(img)  # 类似于上一行，这一行创建了一个布尔掩码，但这次是标识图像中 NaN 的位置。
    img_nan_mask1 = img[non_nan_mask].view(-1, 15)
    processed_img = remove_outliers_percentile_columnwise(img_nan_mask1)
