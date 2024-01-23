import os
import torch.backends.cudnn as cudnn
import torch.utils.data
import tifffile as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from torchvision import transforms


def get_minibatches(inputs, chunksize = 1024 * 32):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #看能否用cuda，否则用cpu

    model_root = os.path.join('..', 'models') #使用 os.path.join 函数，将 '..' 和 'models' 连接起来，以获取模型文件夹的完整路径。
    image_root = os.path.join('..', 'dataset')
    cuda = True #表明是否使用 CUDA 加速。
    cudnn.benchmark = True  #启用了 CuDNN 的 benchmark 功能，该功能会在每次前向传播时优化卷积操作的计算，以提高性能
    lr = 1e-3
    batch_size = 128 #训练时使用的批次大小，即每次模型更新所处理的样本数量。
    image_size = 28 #图像的大小
    epoch = 18
    alpha = 0

    """load data"""
    path = '../dataset/lishu_s12_21.tiff'
    img_tf = tf.imread(path)
    img = torch.tensor(img_tf).view(15, -1).float()
    img = img.T
    #将 NumPy 数组 img_tf 转换为 PyTorch 的张量（tensor），并将其放到指定的 device（CPU 或 GPU）上。
    #.view(-1, 15)的操作将图像数据重新形状为二维张量，其中每行包含15个特征。最后，.float()将数据类型转换为浮点数。

    non_nan_mask = ~torch.isnan(img) #创建了一个布尔掩码（Boolean mask），用于标识图像中非 NaN（Not a Number）的位置。
    # ~ 操作符对这个张量进行按位取反，得到非 NaN 的位置为 True，NaN 的位置为 False。
    nan_mask = torch.isnan(img) # 类似于上一行，这一行创建了一个布尔掩码，但这次是标识图像中 NaN 的位置。
    img_nan_mask1=img[non_nan_mask].view(-1,15)

    """normalization"""
    # 计算均值和标准差，只考虑非NaN值
    mean = torch.mean(img_nan_mask1,dim=0,keepdim=True) #计算图像中非 NaN 像素值的均值。通过索引 img[non_nan_mask]，只考虑非 NaN 的像素值。
    #print(img_nan_mask1.shape)
    std = torch.std(img_nan_mask1,dim=0,keepdim=True) #计算图像中非 NaN 像素值的标准差。
    #print(mean,std)
    normalize = transforms.Lambda(lambda x: (x - mean) / std) #创建一个变换（transformation），这里定义了一个匿名函数，对输入的张量x进行标准化操作
    img = normalize(img) #用上一步定义的标准化函数对图像进行处理。

    """get batch"""
    chunksize = batch_size * 1024 # 定义每个小批次的大小
    batches = get_minibatches(img, chunksize=chunksize)
    # 使用定义的get_minibatches函数（10行），将大张量分割成小批次。batches是一个包含这些小批次的列表。

    """ testing """
    my_net = torch.load(os.path.join( #torch.load 用于加载模型的权重和结构。
        model_root, 'mnist_mnistm_model_epoch_' + str(epoch) + '.pth'
    ))
    my_net = my_net.eval() #将模型设置为评估模式，这通常是在进行推断（测试）时使用的模式。

    if cuda:
        my_net = my_net.cuda() #如果使用 CUDA（GPU），将模型移动到 GPU 上。

    len_batches = len(batches) #计算图像分块后的数量。

    i = 0 #表示当前处理的 batch 的索引，即第几个batch。
    n_total = 0 #总共处理的样本数量。
    n_correct = 0 #模型正确分类的样本数量。
    img_result = [] #存储模型的测试结果。

    for batch in batches: #对于每个分块的数据进行循环遍历。
        # test model using target data
        batch = batch.to(device) #将当前的 batch 移动到指定的计算设备（GPU 或 CPU）上。
        class_output, _ = my_net(input_data=batch, alpha=alpha) #使用神经网络模型my_net对当前的batch进行前向传播，得到分类器的输出class_output。
        pred = class_output.data.max(1, keepdim=True)[1] #通过取输出中概率最大的类别来得到模型的预测结果 pred。
        #从模型的输出class_output中找到每个样本预测的类别。

        img_result.append(pred) #将每个batch的预测结果添加到列表img_result中。

    img_pred = torch.cat(img_result)
    #将存储了每个子批次预测结果的列表 img_result 连接成一个大的张量 img_pred。这个张量中包含了所有样本的预测结果。
    img_pred[nan_mask[:,0]] = -1
    # 根据 nan_mask 中的信息，将其中被标记为 NaN 的位置对应的预测结果设为 -1。nan_mask[:, 0] 是一个布尔张量，用于标识哪些位置是 NaN。
    img_pred = img_pred.view(6941, 8449) #将 img_pred 重新调整为形状为 (6941, 8449) 的张量，以便进行图像的显示。
    colors = ['white', 'yellow', 'green','blue']
    cmap = ListedColormap(colors)
    #cmap = plt.get_cmap('viridis').copy()
    #custom_cmap = LinearSegmentedColormap.from_list('white_to_lightgreen', colors, N=256)
    #cmap.set_under('white')
    plt.imshow(img_pred, cmap=cmap, vmin=-1)
    plt.colorbar()
    plt.show()

    # img_pred = torch.cat(img_result).view(6941, 8449)
    # colors = ['white', 'green', 'blue']
    # cmap = ListedColormap(colors)
    # plt.imshow(img_pred, cmap=cmap)
    # plt.colorbar()
    # plt.show()