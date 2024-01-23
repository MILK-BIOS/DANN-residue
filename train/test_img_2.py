import os
import torch.backends.cudnn as cudnn
import torch.utils.data
import tifffile as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from torchvision import transforms
from matplotlib.colors import ListedColormap
import torch
import numpy as np
import pandas as pd


# 定义 get_minibatches 函数，此处省略，使用你之前定义的函数
def get_minibatches(inputs, chunksize = 1024 * 32):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_root = os.path.join('..', 'models')
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

    non_nan_mask = ~torch.isnan(img)  # 创建了一个布尔掩码（Boolean mask），用于标识图像中非 NaN（Not a Number）的位置。
    # ~ 操作符对这个张量进行按位取反，得到非 NaN 的位置为 True，NaN 的位置为 False。
    nan_mask = torch.isnan(img)  # 类似于上一行，这一行创建了一个布尔掩码，但这次是标识图像中 NaN 的位置。
    img_nan_mask1 = img[non_nan_mask].view(-1, 15)

    """normalization"""
    # 计算均值和标准差，只考虑非NaN值
    # mean = torch.mean(img_nan_mask1, dim=0, keepdim=True)  # 计算图像中非 NaN 像素值的均值。通过索引 img[non_nan_mask]，只考虑非 NaN 的像素值。
    # print(img_nan_mask1.shape)
    # std = torch.std(img_nan_mask1, dim=0, keepdim=True)  # 计算图像中非 NaN 像素值的标准差。
    # print(mean,std)
    if path == '../dataset/lishu_s12_20.tiff':
        mean = np.array([-1.6417e+01, -3.1448e+01,  3.6362e+01,  4.5159e-01,  1.2899e+03,
              7.0286e-01,  7.9835e+02,  1.2558e+03,  1.9334e+03,  2.9812e+03,
              2.6952e+03,  1.4174e-01, -1.7032e-01,  4.4627e-02,  1.3331e+00])
        std = np.array([1.9450e+01, 3.2024e+01, 3.7120e+01, 3.9507e-01, 2.8748e+03, 2.2344e-01,
             1.6949e+02, 2.2510e+02, 3.3402e+02, 4.7317e+02, 2.0759e+02, 2.9258e-02,
             7.6024e-02, 7.4289e-02, 7.8293e-02])
    else:
        mean = np.array([-1.5640e+01, -3.0492e+01,  3.5560e+01,  6.8423e-01,  2.7796e+04,
          5.2921e-01,  8.3131e+02,  1.1490e+03,  1.6269e+03,  2.3394e+03,
          2.7287e+03,  7.9246e-02, -2.5195e-01, -7.6281e-02,  1.1767e+00])
        std = np.array([6.8390e+01, 1.0825e+02, 1.2776e+02, 6.2501e+02, 3.6862e+07, 2.9474e+02,
         1.7381e+02, 2.3112e+02, 3.3107e+02, 4.5758e+02, 5.4512e+02, 4.1461e-02,
         8.5349e-02, 9.1382e-02, 1.0025e-01])
    normalize = transforms.Lambda(lambda x: (x - mean) / std)  # 创建一个变换（transformation），这里定义了一个匿名函数，对输入的张量x进行标准化操作
    img = normalize(img)  # 用上一步定义的标准化函数对图像进行处理。
    img = img.float()
    # 绘制散点图
    # 获取数组的行数
    # num_rows = img.shape[0]
    #
    # # 随机选择1000个索引
    # random_indices = np.random.choice(num_rows, size=1000000, replace=False)
    #
    # # 根据随机索引获取相应行
    # sampled_points = img[random_indices, :]
    # plt.scatter(sampled_points[:, 0], sampled_points[:, 1])
    # plt.title('Scatter Plot of Tensor')
    #
    # # 显示图像
    # plt.show()

    """get batch"""
    chunksize = batch_size * 1024
    batches = get_minibatches(img, chunksize=chunksize)

    # 设置要测试的 epoch 范围
    start_epoch = 26
    end_epoch = 29

    # 创建一个子图，每行显示多少个图像取决于 ncols 参数
    ncols = 2
    nrows = (end_epoch - start_epoch + 1) // ncols + ((end_epoch - start_epoch + 1) % ncols > 0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows))

    # 收集所有的结果
    all_results = []

    # 遍历不同的 epoch 进行测试
    for epoch in range(start_epoch, end_epoch + 1):
        """ testing """
        my_net = torch.load(os.path.join(
            model_root, 'mnist_mnistm_model_epoch_' + str(epoch) + '.pth'
        ))
        my_net = my_net.eval()

        if cuda:
            my_net = my_net.cuda()

        len_batches = len(batches)

        i = 0
        n_total = 0
        n_correct = 0
        img_result = []

        for batch in batches:
            # test model using target data
            batch = batch.to(device)
            class_output, _ = my_net(input_data=batch, alpha=alpha)
            pred = class_output.data.max(1, keepdim=True)[1]

            img_result.append(pred)

        img_pred = torch.cat(img_result)
        img_pred[nan_mask[:, 0]] = -1
        img_pred = img_pred.view(6941, 8449)
        colors = ['white', 'yellow', 'green', 'blue']
        cmap = ListedColormap(colors)

        # 将结果添加到列表中
        all_results.append((img_pred, epoch))

     # 绘制所有图像
    for idx, (img_pred, epoch) in enumerate(all_results):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].imshow(img_pred, cmap=cmap, vmin=-1)
        axes[row, col].set_title(f"Epoch {epoch}")

    # 将子图布局调整为紧凑
    plt.tight_layout()
    plt.show()










