#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: train.py
@datatime: 4/21/2023 10:16 AM
"""

# 导入torch的F
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from mymodels.models import Net
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.BackupCode import *
from mymodels.resnet import resnet18
from dataset.class_divide import get_fold_filelist
from dataset.data_loader import get_loader_difficult
from utils.tictoc import TicToc
from mymodels.unetr import UNETR
from mymodels.Unet import UNet
import utils.evaluation as ue
from utils.myloss import SoftDiceLoss, JaccardLoss
import test

import warnings
warnings.filterwarnings('ignore', message='Argument \'interpolation\' of type int is deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.')

sep = os.sep  # os.sep根据你所处的平台，自动采用相应的分隔符号


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return param_size, param_sum, buffer_size, buffer_sum, all_size


def mnist_loader():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        lambda x: torch.stack([x[0], x[0], x[0]], dim=0)
    ])
    # 加载测试数据集
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

    # 划分数据集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    indices = list(range(len(train_dataset)))
    train_idx, val_idx, test_idx = indices[:50000], indices[50000:60000], indices[60000:]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4096, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=1024, sampler=val_sampler)
    return train_loader, val_loader, test_loader


def getdataset(csv_file, fold_K, fold_idx, image_size, batch_size, num_workers):
    augmentation_prob = 1.0
    train, valid, test = get_fold_filelist(csv_file, K=fold_K, fold=fold_idx, validation=True)

    # 输出 train, valid, test的大小
    print('train size: ', len(train))
    print('valid size: ', len(valid))
    print('test size: ', len(test))

    # 将train的每个元素的第二个元素转换为int，并保存到一个新的列表中
    train_class_list = [i[1] for i in train]
    train_class_list = [int(i) for i in train_class_list]
    # 输出train_class_list
    print('train_class_list: ', train_class_list)

    # 将valid的每个元素的第二个元素转换为int，并保存到一个新的列表中
    valid_class_list = [i[1] for i in valid]
    valid_class_list = [int(i) for i in valid_class_list]
    # 输出valid_class_list
    print('valid_class_list: ', valid_class_list)

    # 将test的每个元素的第二个元素转换为int，并保存到一个新的列表中
    test_class_list = [i[1] for i in test]
    test_class_list = [int(i) for i in test_class_list]
    # 输出test_class_list
    print('test_class_list: ', test_class_list)




    filepath_img = './class_out/stage1/p_image'
    filepath_mask = './class_out/stage1/p_mask'
    filepath_contour = './class_out/stage1/p_contour'
    filepath_dist = './class_out/stage1/p_distance_D1'

    # 将train, valid, test保存到一个txt文件中
    train_fold_txt = './foldinfo/train_fold_' + str(fold_idx) + '.txt'
    valid_fold_txt = './foldinfo/valid_fold_' + str(fold_idx) + '.txt'
    test_fold_txt = './foldinfo/test_fold_' + str(fold_idx) + '.txt'
    with open(train_fold_txt, 'w') as f:
        for i in train:
            f.write(i[0] + '\n')
    with open(valid_fold_txt, 'w') as f:
        for i in valid:
            f.write(i[0] + '\n')
    with open(test_fold_txt, 'w') as f:
        for i in test:
            f.write(i[0] + '\n')

    train_list = [filepath_img + sep + i[0] for i in train]
    train_list_GT = [filepath_mask + sep + i[0] for i in train]
    trian_list_contour = [filepath_contour + sep + i[0] for i in train]
    train_list_dist = [filepath_dist + sep + i[0] for i in train]
    train_class_list_GT = [i[1] for i in train]

    valid_list = [filepath_img + sep + i[0] for i in valid]
    valid_list_GT = [filepath_mask + sep + i[0] for i in valid]
    valid_list_contour = [filepath_contour + sep + i[0] for i in valid]
    valid_list_dist = [filepath_dist + sep + i[0] for i in valid]
    valid_class_list_GT = [i[1] for i in valid]

    test_list = [filepath_img + sep + i[0] for i in test]
    test_list_GT = [filepath_mask + sep + i[0] for i in test]
    test_list_contour = [filepath_contour + sep + i[0] for i in test]
    test_list_dist = [filepath_dist + sep + i[0] for i in test]
    test_class_list_GT = [i[1] for i in test]

    print("images count in train:{}".format(len(train_list)))
    print("images count in valid:{}".format(len(valid_list)))
    print("images count in test :{}".format(len(test_list)))


    train_loader = get_loader_difficult(seg_list=None,
                                        GT_list=train_list_GT,
                                        class_list=train_class_list_GT,
                                        image_list=train_list,
                                        contour_list=trian_list_contour,
                                        dist_list=train_list_dist,
                                        image_size=image_size,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        mode='train',
                                        augmentation_prob=augmentation_prob, )

    valid_loader = get_loader_difficult(seg_list=None,
                                        GT_list=valid_list_GT,
                                        class_list=valid_class_list_GT,
                                        image_list=valid_list,
                                        contour_list=valid_list_contour,
                                        dist_list=valid_list_dist,
                                        image_size=image_size,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        mode='val',
                                        augmentation_prob=0., )

    test_loader = get_loader_difficult(seg_list=None,
                                        GT_list=test_list_GT,
                                        class_list=test_class_list_GT,
                                        image_list=test_list,
                                        contour_list=test_list_contour,
                                        dist_list=test_list_dist,
                                        image_size=image_size,
                                        batch_size=4,
                                        num_workers=num_workers,
                                        mode='test',
                                        augmentation_prob=0., )

    return train_loader, valid_loader, test_loader


def breast_loader(batch_size):
    train_path_m = './train_path/fold/fold'
    csv_path = './class_out/train.csv'
    fold_k = 5
    fold_idx = 1
    fold_id = 1
    distance_type = "dist_mask"
    normal_flag = False
    image_size = 256
    num_workers = 5

    print('batch_size: ', batch_size)
    train_loader, valid_loader, test_loader = getdataset(csv_path, fold_k, fold_idx, image_size, batch_size, num_workers)

    # train_path = train_path_m + str(fold_id) + '/train/images/'  # train_path是指训练集图片路径
    # val_path = train_path_m + str(fold_id) + '/validation/images/'  # val_path是指验证集图片路径
    # test_path = train_path_m + str(fold_id) + '/test/images/'  # test_path是指测试集图片路径
    # train_file_names = glob.glob(train_path + "*.png")  # 获取训练集图片路径
    #
    # # 为了避免模型只记住了数据的顺序，而非真正的特征，代码使用了 random.shuffle() 函数对 train_file_names
    # # 变量中存储的图片路径进行了随机打乱操作，从而增加了数据的随机性，更有助于训练出鲁棒性更强的模型。
    # random.shuffle(train_file_names)  # 打乱训练集图片路径
    # val_file_names = glob.glob(val_path + "*.png")  # 获取验证集图片路径
    # test_file_names = glob.glob(test_path + "*.png")  # 获取测试集图片路径
    #
    # # todo: add TTA here
    #
    # trainLoader = DataLoader(
    #     DatasetImageMaskContourDist(train_file_names, distance_type, normal_flag),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=5,
    # )
    # validLoader = DataLoader(
    #     DatasetImageMaskContourDist(val_file_names, distance_type, normal_flag),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=5,
    # )
    # testLoader = DataLoader(
    #     DatasetImageMaskContourDist(test_file_names, distance_type, normal_flag),
    #     num_workers=1,
    #     batch_size=10,
    #     shuffle=True,
    # )

    return train_loader, valid_loader, test_loader
def Train_breast():
    project = 'UNet2_0'   # project name-----------------------------------------------------
    epoch_num = 100     # epoch_num -----------------------------------------------------
    lr = 0.0005  # 学习率 -----------------------------------------------------
    bs = 5  # batch_size -----------------------------------------------------
    L = 0.2  # 代表的是seg_loss的权重 -----------------------------------------------------
    use_pretrained = False  # 是否使用预训练模型 -----------------------------------------------------
    model = UNet(1, 1)     # -----------------------------------------------------
    log_dir = './log/log'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dir = './savemodel'
    save_model_dir = os.path.join(model_dir, project)
    t = TicToc()
    te = TicToc()
    content = "per epoch Time: "
    contentvalid = "per epoch training&vlidation test Time: "
    contentwholeepoch = "whole epoch Time: "
    contenttotal = "total cost: "
    is_train = True
    is_test = True  # False
    is_continue_train = False

    print(getModelSize(model))

    # todo: functionize it
    if torch.cuda.is_available():
        print("Using GPU")
        model = DataParallel(model)
        model.to(device)
    else:
        print("Using CPU")
        # model = DataParallel(model)
        model.to(device)

    train_loader, valid_loader, test_loader = breast_loader(bs)

    criterion_cls = nn.NLLLoss()    # -----------------------------------------------------
    # criterion_cls = nn.CrossEntropyLoss()    # -----------------------------------------------------
    criterion_seg = SoftDiceLoss()    # -----------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr)   # -----------------------------------------------------

    if is_continue_train:
        model.load_state_dict(torch.load('model.pth'))
        print('load model')

    if is_train:
        temploss = 100.0
        tempacc = 0.4
        Iter = 0

        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        else:
            # 删掉原来的model文件
            shutil.rmtree(save_model_dir)
            os.makedirs(save_model_dir)
        log_dir = os.path.join(log_dir, project)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            # 删掉原来的log文件
            shutil.rmtree(log_dir)
            os.makedirs(log_dir)

        writer = SummaryWriter(log_dir=log_dir)
        datas = train_loader     # -----------------------------------------------------
        for epoch in range(epoch_num):
            t.ticbegin()
            te.ticbegin()
            cls_running_loss = 0.0
            seg_running_loss = 0.0
            running_loss = 0.0  # running_loss是所有batch的loss之和
            print('epoch: %d / %d' % (epoch + 1, epoch_num))
            for i, data in tqdm(enumerate(datas, 0), total=len(datas)):
                (img_file_name, inputs, targets1, targets2, targets3, targets4) = data
                if torch.cuda.is_available():
                    inputs = inputs.to(device)
                    targets1 = targets1.to(device)
                    targets4 = targets4.to(device)

                optimizer.zero_grad()
                segout, outputs = model(inputs)
                seg_loss = criterion_seg(segout, targets1)
                cls_loss = criterion_cls(outputs, targets4)
                loss = L * seg_loss + (1 - L) * cls_loss
                loss.backward()
                optimizer.step()

                cls_running_loss += cls_loss.item()  # loss.item()是一个batch的loss, running_loss是所有batch的loss之和
                seg_running_loss += seg_loss.item()  # loss.item()是一个batch的loss, running_loss是所有batch的loss之和
                running_loss += loss.item()  # loss.item()是一个batch的loss, running_loss是所有batch的loss之和
                Iter += 1
                writer.add_scalars('Loss', {'cls_running_loss': cls_running_loss}, Iter)
                writer.add_scalars('Loss', {'seg_running_loss': seg_running_loss}, Iter)
                writer.add_scalars('Loss', {'running_loss': running_loss}, Iter)

            # 计时结束
            t.ticend()
            t.printtime(content)
            t.ticbegin()

            # todo:simplize it
            # 计算平均epoch_cls_loss
            epoch_cls_loss = cls_running_loss / len(datas)  # len(train_loader)是batch的个数---------------
            writer.add_scalars('Loss', {'epoch_cls_loss': epoch_cls_loss}, epoch)
            epoch_seg_loss = seg_running_loss / len(datas)  # len(train_loader)是batch的个数-----------------
            writer.add_scalars('Loss', {'epoch_seg_loss': epoch_seg_loss}, epoch)
            epoch_loss = running_loss / len(datas)  # len(train_loader)是batch的个数-----------------
            writer.add_scalars('Loss', {'epoch_loss': epoch_loss}, epoch)

            print('epoch_cls_loss = ', epoch_cls_loss)
            print('epoch_seg_loss = ', epoch_seg_loss)
            print('epoch_loss = ', epoch_loss, '\n')

            # 保存模型策略
            if epoch % 10 == 0:  # 每10个epoch保存一次模型
                torch.save(model.state_dict(), save_model_dir + '/model' + str(epoch) + '.pth')
                print('save model')
            if temploss > epoch_cls_loss:
                temploss = epoch_cls_loss
                torch.save(model.state_dict(), save_model_dir + '/miniclsloss' + '.pth')
                print('save model，and epoch_cls_loss = ', temploss, '\n')

            print('Iter = ', Iter)
            if epoch % 3 == 0:
                test.trainvalid('train', datas, model, device, writer, Iter)
                test.trainvalid('valid', valid_loader, model, device, writer, Iter)

            t.ticend()
            t.printtime(contentvalid)
            te.ticend()
            te.printtime(contentwholeepoch)
            te.printlefttime(epoch, epoch_num)
        t.printtime(contenttotal, True)
        writer.close()

        torch.save(model.state_dict(), 'model.pth')


    print('Finished Training\n')
    if is_test:
        mini_loss_model = save_model_dir + '/miniclsloss' + '.pth'
        model.load_state_dict(torch.load(mini_loss_model))
        test.test('test', test_loader, model, device)

    print('\nFinished Testing\n')


def Train_Mnist():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    print(getModelSize(model))

    if torch.cuda.is_available():
        print("Using GPU")
        model = DataParallel(model)
        model.to(device)
    else:
        print("Using CPU")
        # model = DataParallel(model)
        model.to(device)

    train_loader, val_loader, test_loader = mnist_loader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.99)

    is_train = True
    is_test = True
    is_continue_train = False

    if is_continue_train:
        model.load_state_dict(torch.load('model.pth'))
        print('load model')

    if is_train:
        for epoch in range(100):
            running_loss = 0.0
            for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))  # i + 1代表第几个batch
                    running_loss = 0.0
                if i == 0 and epoch % 5 == 0:
                    torch.save(model.state_dict(), 'model.pth')
                    print('save model')

        torch.save(model.state_dict(), 'model.pth')

    model.load_state_dict(torch.load('model.pth'))

    if is_test:
        # 训练集上测试
        correct = 0
        total = 0
        with torch.no_grad():
            print('train set testing...')
            for data in train_loader:
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        print('Accuracy of the network on the 50000 train images: %.4f %%' % (100 * correct / total))


        print('testing...')
        # 测试模型
        correct = 0
        total = 0
        with torch.no_grad():
            print('testing...')
            for i, data in enumerate(test_loader , 0):
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # print('测试集上的准确率: %.3f %%' % (100 * correct / total))

        # 计算准确率
        accuracy = correct / total
        print('Accuracy of the network on the test images: %.4f %%' % (100 * accuracy))

if __name__ == '__main__':
    # Train_Mnist()
    Train_breast()


