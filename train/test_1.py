import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from dataset.data_loader_2 import load_data
from torchvision import datasets


def test(dataset_name, epoch):
    assert dataset_name in ['lishu_s12_21_samples.csv', 'lishu_s12_20_samples.csv']

    model_root = os.path.join('..', 'models') #使用 os.path.join 方法创建了一个文件路径，将 '..'（上一级目录）和 'models' 连接在一起
    image_root = os.path.join('..', 'dataset') #和‘dataset’连接在一起

    cuda = True #在可用的情况下将使用 CUDA 加速
    cudnn.benchmark = True #启用了 PyTorch 中 cuDNN（CUDA Deep Neural Network library）的自动优化
    batch_size = 128 #在模型测试中每个批次的样本数
    image_size = 28 #表示图像的尺寸
    alpha = 0

    """load data"""
    if dataset_name == 'lishu_s12_20_samples.csv':

        # 调用 load_data 函数加载数据集，传入数据集根目录、数据集名称和 transform 标志
        dataset = load_data(
            data_root=os.path.join(image_root, dataset_name),#将 image_root（图像数据的根目录）和 dataset_name（数据集名称）拼接在一起，形成数据集的完整路径
            data_set=dataset_name,
            transform=True,
            domain_flag=1
        )
    else:
        dataset = load_data(
            data_root=os.path.join(image_root, dataset_name),
            data_set=dataset_name,
            transform=True,
            domain_flag=0
        )

    # 创建一个 PyTorch 的 DataLoader，用于批量加载数据
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,  # 数据集
        batch_size=batch_size,  # 每个批次的样本数
        shuffle=False,  # 不打乱数据顺序
        num_workers=2  # 使用的数据加载器的工作线程数
    )


    """ training """

    my_net = torch.load(os.path.join(
        model_root, 'mnist_mnistm_model_epoch_' + str(epoch) + '.pth'
    ))#通过使用 torch.load 函数，从指定路径加载了PyTorch模型文件。这个文件包含了在训练过程中保存的神经网络模型的权重和参数。
    my_net = my_net.eval()#将加载的神经网络模型设置为评估模式

    if cuda:#如果 cuda 标志为 True，表示使用 GPU 加速，将神经网络模型移动到 GPU 上进行计算。这可以加速模型的推理过程。
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)#获取数据加载器 dataloader 的长度，即数据集的批次数量。这个值将用于迭代数据集。
    data_target_iter = iter(dataloader)#创建了数据加载器的迭代器，用于逐批次地获取数据。

    i = 0#初始化变量 i 为 0，这是用于迭代器索引的计数器。
    n_total = 0#初始化 n_total 和 n_correct 为 0，这两个变量将用于统计总样本数和正确预测的样本数。
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = next(data_target_iter) #使用 next 函数从目标数据集的迭代器 data_target_iter 中获取下一个批次的数据。
        t_img, t_label = data_target #data_target 是一个包含图像数据 t_img 和标签数据 t_label 的元组。

        batch_size = len(t_label) #获取当前批次的样本数量，即标签数据 t_label 的长度。

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        #其中 batch_size 是当前批次的样本数量，3 表示图像通道数（假设是 RGB 彩色图像），image_size 是图像的尺寸。
        class_label = torch.LongTensor(batch_size)
        #创建一个长整型张量，用于存储类别标签数据。张量的形状为 (batch_size)，其中 batch_size 是当前批次的样本数量。

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        #调整 input_img 张量的大小以匹配 t_img 张量的大小。这样确保它们具有相同的形状。将 t_img 张量的值复制到已经调整大小的 input_img 张量中。
        class_label.resize_as_(t_label).copy_(t_label)

        class_output, _ = my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        ##print(pred)
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
