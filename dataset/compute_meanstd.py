import os
import torch.backends.cudnn as cudnn
import torch.utils.data
import tifffile as tf
import torch
import numpy as np


def remove_outliers_percentile_columnwise(tensor, lower_percentile=0, upper_percentile=100):
    # 将张量转换为 NumPy 数组
    numpy_tensor = tensor.numpy()

    # 按列分开张量
    columns = torch.split(tensor, 1, dim=1)

    # 对每列进行异常值处理
    processed_columns = [remove_outliers_percentile(column.squeeze().numpy(), lower_percentile, upper_percentile)
                         for column in columns]

    # 将处理后的列重新堆叠成张量
    processed_tensor = torch.stack(processed_columns, dim=1)

    return processed_tensor

def remove_outliers_percentile(column, lower_percentile=3, upper_percentile=97):
    lower_bound = np.percentile(column, lower_percentile)
    upper_bound = np.percentile(column, upper_percentile)

    # 将异常值限制在边界内
    processed_column = np.clip(column, lower_bound, upper_bound)

    return torch.tensor(processed_column, dtype=torch.float32)



def compute_meanstd(data):
    mean = torch.mean(data, dim=0, keepdim=True)  # 计算图像中非 NaN 像素值的均值。通过索引 img[non_nan_mask]，只考虑非 NaN 的像素值。
    print(mean)
    std = torch.std(data, dim=0, keepdim=True)  # 计算图像中非 NaN 像素值的标准差。
    print(std)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_root = os.path.join('', 'models')
    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""
    path = '../dataset/lishu_s12_21.tiff'
    img_tf = tf.imread(path)
    img = torch.tensor(img_tf).view(15, -1).float()
    img = img.T
    print(img.shape)

    non_nan_mask = ~torch.isnan(img)  # 创建了一个布尔掩码（Boolean mask），用于标识图像中非 NaN（Not a Number）的位置。
    # ~ 操作符对这个张量进行按位取反，得到非 NaN 的位置为 True，NaN 的位置为 False。
    nan_mask = torch.isnan(img)  # 类似于上一行，这一行创建了一个布尔掩码，但这次是标识图像中 NaN 的位置。
    img_nan_mask1 = img[non_nan_mask].view(-1, 15)
    processed_img = remove_outliers_percentile_columnwise(img_nan_mask1)
    print(processed_img.shape)
    compute_meanstd(processed_img)

