import os
import torch
import torch.utils.data
import tifffile as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def get_minibatches(inputs, chunksize=1024 * 32):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


if __name__ == '__main__':
    device = torch.device("cpu")

    model_root = os.path.join('..', 'models')
    image_root = os.path.join('..', 'dataset')
    lr = 1e-3
    batch_size = 4
    image_size = 28
    epoch = 5
    alpha = 0

    """load data"""
    path = '../dataset/lishu_s12_21.tiff'
    img_tf = tf.imread(path)
    img = torch.tensor(img_tf, device=device).view(15, -1).float()
    img = img.T
    non_nan_mask = ~torch.isnan(img)
    nan_mask = torch.isnan(img)

    """normalization"""
    '''
    # 计算均值和标准差，只考虑非 NaN 值
    mean = torch.mean(img[non_nan_mask])
    std = torch.std(img[non_nan_mask])

    # 这里的 transforms.Lambda 可能在 CPU 上不被支持，手动进行标准化
    img = (img - mean) / std
    '''
    """get batch"""
    chunksize = batch_size * 1024
    batches = get_minibatches(img, chunksize=chunksize)

    """ testing """
    my_net = torch.load(os.path.join(
        model_root, 'mnist_mnistm_model_epoch_' + str(epoch) + '.pth'
    ), map_location='cpu')
    my_net = my_net.eval()

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
    plt.imshow(img_pred, cmap=cmap, vmin=-1)
    plt.colorbar()
    plt.show()
