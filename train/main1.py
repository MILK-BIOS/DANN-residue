import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataset.data_loader_2 import load_data
from torchvision import datasets
from models.FC_model import FCModel
import numpy as np
from test_1 import test


if __name__ == '__main__':
    source_dataset_name = 'lishu_s12_21_samples.csv'
    target_dataset_name = 'lishu_2020_random_points_1.csv'
    target_valid_dataset_name = 'lishu_s12_20_samples.csv'
    source_data_root = os.path.join('..', 'dataset', source_dataset_name)
    target_data_root = os.path.join('..', 'dataset', target_dataset_name)
    model_root = os.path.join('..', 'models')
    cuda = True
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 128
    n_epoch = 100

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    print('init finished!')

    # load data
    dataset_source = load_data(
        data_root=source_data_root,
        data_set=source_dataset_name,
        transform=True,  #指定是否应用数据变换
        domain_flag=0   #指定数据集的域标识，这里设为0
    )

       #创建了一个名为dataloader_source的数据加载器对象，等号后面没看懂
    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,   #要加载的数据集对象
        batch_size=batch_size,   #批次
        shuffle=True,   #表示是否在每个迭代中对数据进行洗牌，打乱数据的顺序，这样可以增加模型的泛化性能
        num_workers=2)   #用于数据加载的子进程数目，可以加速数据加载过程


    dataset_target = load_data(
        data_root=target_data_root,
        data_set=target_dataset_name,
        transform=True,
        domain_flag=1
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)
    print('data load finished!')
    # load model

    my_net = FCModel()   #调用FCModel

    # setup optimizer  模型的优化和损失函数的定义
    # 创建了一个Adam优化器，，用于更新神经网络模型参数以最小化训练中的损失
    optimizer = optim.Adam(my_net.parameters(), lr=lr)
    #创建了负对数似然损失（Negative Log Likelihood Loss）
    loss_class = torch.nn.NLLLoss()   #度量分类任务中的损失
    loss_domain = torch.nn.NLLLoss()   #度量域适应任务中的损失

    if cuda:
        my_net = my_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in my_net.parameters():   #遍历神经网络模型my_net中的所有参数
        p.requires_grad = True   #将每个参数的requires_grad属性设置为True，表示这些参数需要计算梯度，从而可以在训练过程中进行更新。
    print('model init finished!')
    # training
    print('---------------------training---------------------')
    for epoch in range(n_epoch):

        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        i = 0
        while i < len_dataloader:

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            try:
                data_source = next(data_source_iter)
            except Exception as e:
                print(f"Error while loading data from the source: {e}")
                raise  # 将异常再次抛出，以中断程序执行并打印完整的错误信息
            s_img, s_label = data_source

            my_net.zero_grad()
            batch_size = len(s_label)

            input_img = torch.FloatTensor(batch_size, 15)
            class_label = torch.LongTensor(batch_size)
            domain_label = torch.zeros(batch_size)
            domain_label = domain_label.long()

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                input_img = input_img.cuda()
                class_label = class_label.cuda()
                domain_label = domain_label.cuda()

            input_img.resize_as_(s_img).copy_(s_img)
            class_label.resize_as_(s_label).copy_(s_label)
            class_output, domain_output = my_net(input_data=input_img, alpha=alpha)

            err_s_label = loss_class(class_output, class_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # training model using target data
            data_target = next(data_target_iter)
            t_img, _ = data_target

            batch_size = len(t_img)

            input_img = torch.FloatTensor(batch_size, 15)
            domain_label = torch.ones(batch_size)
            domain_label = domain_label.long()

            if cuda:
                t_img = t_img.cuda()
                input_img = input_img.cuda()
                domain_label = domain_label.cuda()

            input_img.resize_as_(t_img).copy_(t_img)

            _, domain_output = my_net(input_data=input_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()

            i += 1

            print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                  % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                     err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

        torch.save(my_net, '{0}/mnist_mnistm_model_epoch_{1}.pth'.format(model_root, epoch))
        test(source_dataset_name, epoch)
        test(target_valid_dataset_name, epoch)

    print('done')
