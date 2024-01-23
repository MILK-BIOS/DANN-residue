import os
import torch.backends.cudnn as cudnn
import torch.utils.data
import tifffile as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def get_minibatches(inputs, chunksize = 1024 * 32):
    #这是一个函数定义，函数名为 get_minibatches，接受两个参数：inputs 是要被分割的大张量，chunksize 是每个小批次的大小，默认为 1024 * 32。
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]
    # 使用列表推导式，通过循环遍历inputs张量，以大小为chunksize的步长进行切片，形成小批次的列表。
    # 返回的列表中的每个元素都是输入数据的一个小批次。这个函数的目的是将大张量分割成一系列的小批次，便于进行分批处理。

if __name__ == '__main__': #表示当脚本作为主程序运行时执行以下代码块。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_dataset_name = 'MNIST' #源数据集的名称和路径
    target_dataset_name = 'mnist_m' #目标域数据集名称
    source_image_root = os.path.join('..', 'dataset', source_dataset_name)
    target_image_root = os.path.join('..', 'dataset', target_dataset_name)
    model_root = os.path.join('..', 'models')
    cuda = True
    cudnn.benchmark = True
    lr = 1e-3 # 学习率
    batch_size = 128
    image_size = 28
    epoch = 31

    model_root = os.path.join('..', 'models')
    image_root = os.path.join('..', 'dataset')

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""
    """path = '../dataset/lishu_sar_residue21.tif'"""
    path = '../dataset/lishu_sar_residue20.tif'
    img_tf = tf.imread(path) #使用 tifffile 库的 imread 函数加载图像数据，img_tf 是一个 NumPy 数组。
    img = torch.tensor(img_tf, device=device).view(-1, 5).float()
    #将图像数据转换为 PyTorch 张量，并移动到指定的设备上（GPU 或 CPU）
    #将NumPy数组img_tf转换为PyTorch张量img。.view(-1, 5)表示将张量的形状调整为二维，每行有5个元素。.float()将数据类型转换为浮点型。

    """get batch"""
    chunksize = batch_size * 1024 # 定义每个小批次的大小
    batches = get_minibatches(img, chunksize=chunksize)
    # 使用定义的get_minibatches函数（9行），将大张量分割成小批次。batches是一个包含这些小批次的列表。

    """ training """
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

    for batch in batches: #遍历小批次数据
        # test model using target data
        batch = batch.to(device) #将小批次数据移动到指定设备上（GPU 或 CPU）
        class_output, _ = my_net(input_data=batch, alpha=alpha)
        #使用加载的神经网络模型 my_net 进行推理（前向计算），输入数据为当前小批次batch，alpha是之前定义的超参数。
        pred = class_output.data.max(1, keepdim=True)[1]
        #获取模型预测的类别索引，类似于之前解释的代码。返回模型输出中每行最大值对应的索引。

        img_result.append(pred)
        #将当前小批次的预测结果添加到img_result列表中。img_result是一个保存了所有小批次预测结果的列表，它将在后续被用于生成最终的图像结果。

    #img_pred = torch.cat(img_result).view(6941, 8449)
    #colors = ['white', 'green', 'blue']
    #cmap = ListedColormap(colors)
    #plt.imshow(img_pred, cmap=cmap)
    #plt.colorbar()
    #plt.show()

    img_pred = torch.cat(img_result)
    #将存储在 img_result 列表中的所有小批次的预测结果拼接成一个大的张量 img_pred。这个张量的形状将是 (总样本数,)。
    img_pred[nan_mask[:,0]] = -1
    #根据 nan_mask 中的信息，将其中被标记为 NaN 的位置对应的预测结果设为 -1。nan_mask[:, 0] 是一个布尔张量，用于标识哪些位置是 NaN。
    img_pred = img_pred.view(6941, 8449)
    colors = ['white', 'yellow', 'green','blue']
    cmap = ListedColormap(colors)
    #cmap = plt.get_cmap('viridis').copy()
    #custom_cmap = LinearSegmentedColormap.from_list('white_to_lightgreen', colors, N=256)
    #cmap.set_under('white')
    plt.imshow(img_pred, cmap=cmap, vmin=-1)
    plt.colorbar()
    plt.show()