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
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.BackupCode import *
from dataset.class_divide import get_fold_filelist
from dataset.data_loader import get_loader_difficult
from utils.tictoc import TicToc
import utils.evaluation as ue
from utils.myloss import SoftDiceLoss, JaccardLoss
import test
from utils import utils
from mymodels import OpenDataSet

import warnings
torch.cuda.empty_cache()
warnings.filterwarnings('ignore',
                        message='Argument \'interpolation\' of type int is deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.')

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


def getdataset(csv_file, fold_K, fold_idx, image_size, batch_size, testbs, num_workers, validate_flag=True):
    augmentation_prob = 0.0
    if validate_flag:
        train, valid, test = get_fold_filelist(csv_file, K=fold_K, fold=fold_idx, validation=True)
    else:
        train, test = get_fold_filelist(csv_file, K=fold_K, fold=fold_idx)
        valid = test  # 为了保持代码的一致性，这里将valid设置为test

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

    # 将train_list, valid_list, test_list保存到一个txt文件中
    train_list_txt = './foldinfo/train_list_' + str(fold_idx) + '.txt'
    valid_list_txt = './foldinfo/valid_list_' + str(fold_idx) + '.txt'
    test_list_txt = './foldinfo/test_list_' + str(fold_idx) + '.txt'
    utils.WriteIntoTxt(train_list, train_list_txt)
    utils.WriteIntoTxt(valid_list, valid_list_txt)
    utils.WriteIntoTxt(test_list, test_list_txt)

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
                                       batch_size=testbs,
                                       num_workers=num_workers,
                                       mode='test',
                                       augmentation_prob=0., )

    return train_loader, valid_loader, test_loader


def breast_loader(batch_size, testbs, validate_flag):
    train_path_m = './train_path/fold/fold'
    csv_path = './class_out/train.csv'
    fold_k = 5
    fold_idx = 1
    fold_id = 1
    distance_type = "dist_mask"
    normal_flag = False
    image_size = 256
    num_workers = 6

    print('batch_size: ', batch_size)
    train_loader, valid_loader, test_loader = getdataset(csv_path, fold_k, fold_idx, image_size, batch_size, testbs,
                                                         num_workers, validate_flag)

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


def Train_breast(Project, Bs, epoch, Model_name, lr, Use_pretrained, _have_segtask, _only_segtask):
    project = Project  # project name-----------------------------------------------------
    epoch_num = epoch  # epoch_num -----------------------------------------------------
    class_num = 1  # class_num -----------------------------------------------------
    lr = lr  # 学习率  -----------------------------------------------------
    validate_flag = False  # 是否使用验证集 -----------------------------------------------------
    lr_low = 1e-12  # 学习率下限  ------------------------------------------------------
    lr_warm_epoch = 5  # warm up 的 epoch 数 -----------------------------------------------------
    lr_cos_epoch = 490  # 学习率下降的epoch数 -----------------------------------------------------
    num_epochs_decay = 10  # 学习率下降的epoch数 -----------------------------------------------------
    decay_step = 10  # 学习率下降的epoch数 -----------------------------------------------------
    decay_ratio = 0.01  # 学习率下降的比例 -----------------------------------------------------
    bs = Bs  # batch_size -----------------------------------------------------
    testbs = Bs  # test_batch_size -----------------------------------------------------
    L = 0.5  # 代表的是seg_loss的权重 -----------------------------------------------------
    use_pretrained = Use_pretrained  # 是否使用预训练模型 -----------------------------------------------------
    model_name = Model_name  # 模型名字 -----------------------------------------------------
    log_dir = './log/log'
    model_dir = './savemodel'
    save_model_dir = os.path.join(model_dir, project)
    t = TicToc()
    te = TicToc()
    content = "----per epoch Time: "
    contentvalid = "----per epoch training&vlidation test Time: "
    contentwholeepoch = "----whole epoch Time: "
    contenttotal = "----total cost: "
    is_train = True
    is_test = True  # False
    is_continue_train = False
    _only_segtask = _only_segtask
    if _only_segtask:
        _have_segtask = True
    else:
        _have_segtask = _have_segtask

    model = utils.InitModel(model_name, use_pretrained, class_num, _have_segtask, _only_segtask)  # ---------------------------------------------
    utils.init_weights(model)

    print(getModelSize(model))
    print('project: ', project)

    model, device = utils.Device(model)
    print(device)
    train_loader, valid_loader, test_loader = breast_loader(bs, testbs, validate_flag)
    # train_loader, test_loader = OpenDataSet.SelectDataSet('Cifar_10', bs)

    # criterion_cls = nn.NLLLoss()    # -----------------------------------------------------
    pos_weight = torch.tensor([515 / 108]).to(device)
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion_seg = SoftDiceLoss()  # -----------------------------------------------------
    criterion_seg = nn.BCELoss()  # -----------------------------------------------------
    optimizer = optim.Adam(list(model.parameters()), lr, (0.5, 0.99))  # ------------------------------------------
    lr_sch = utils.LrDecay(lr_warm_epoch, lr_cos_epoch, lr, lr_low, optimizer)  # -------------------------------

    if is_continue_train:
        model_dir = './savemodel/' + project + '/miniclsloss.pth'
        model.load_state_dict(torch.load(model_dir))
        print('load model')

    if is_train:
        SElist = []
        PClist = []
        F1list = []
        JSlist = []
        DClist = []
        IOUlist = []
        Acclist = []
        torch.autograd.set_detect_anomaly(True)
        Iter = seg_loss = 0
        tmp_pre = tmp_tar = None
        utils.Mkdir(save_model_dir)
        log_dir = os.path.join(log_dir, project)
        utils.Mkdir(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        datas = train_loader  # -----------------------------------------------------
        for epoch in range(epoch_num):
            t.ticbegin()
            te.ticbegin()
            cls_running_loss = 0.0
            seg_running_loss = 0.0
            running_loss = 0.0  # running_loss是所有batch的loss之和
            print('epoch: %d / %d' % (epoch + 1, epoch_num))
            print('current lr:', utils.GetCurrentLr(optimizer))
            num_zero = 0
            num_one = 0
            epoch_tp, epoch_fp, epoch_tn, epoch_fn = 0, 0, 0, 0
            model.train()
            for i, data in tqdm(enumerate(datas, 0), total=len(datas)):
                (img_file_name, inputs, targets1, targets2, targets3, targets4) = data
                optimizer.zero_grad()
                Iter += 1
                # (inputs, targets4) = data
                if class_num <= 2:
                    # 将标签进行修改
                    targets4[targets4 == 0] = 0
                    targets4[targets4 == 1] = 0
                    targets4[targets4 == 2] = 1

                    # 将这里面的所有标签为0的统计个数，存在num_zero中
                    num_zero += (targets4 == 0).sum().item()
                    num_one += (targets4 == 1).sum().item()

                if torch.cuda.is_available():
                    inputs = inputs.to(device)
                    targets4 = targets4.to(device)
                    if _have_segtask:
                        targets1 = targets1.to(device)
                if _only_segtask:
                    segout = model(inputs)
                    segout = torch.sigmoid(segout)
                    SR_flat = segout.view(segout.size(0), -1)
                    GT_flat = targets1.view(targets1.size(0), -1)
                    loss = criterion_seg(segout, targets1)
                    seg_running_loss += loss
                    SE, PC, F1, JS, DC, IOU, Acc = ue.get_all_seg(segout, targets1, device)
                    # 将这些指标存到一个list里面，方便后面计算平均值
                    SElist.append(SE)
                    PClist.append(PC)
                    F1list.append(F1)
                    JSlist.append(JS)
                    DClist.append(DC)
                    IOUlist.append(IOU)
                    Acclist.append(Acc)

                    loss.backward()
                    optimizer.step()
                else:
                    if _have_segtask:
                        outputs, segout = model(inputs)
                        segout = torch.sigmoid(segout)
                        seg_loss = criterion_seg(segout, targets1)
                        seg_running_loss += seg_loss.item()
                    else:
                        outputs = model(inputs)

                    if class_num > 2:
                        labels = F.softmax(outputs, dim=1)  # -----------------------------------------------------
                        _, predicted = torch.max(labels.data, 1)
                        cls_loss = criterion_cls(outputs, targets4)
                    else:  # 如果是二分类，就用sigmoid
                        labels = torch.sigmoid(outputs)
                        predicted = torch.round(labels)
                        targets4v = targets4.view(-1, 1)
                        targets4v = targets4v.to(torch.float)
                        cls_loss = criterion_cls(outputs, targets4v)
                    if _have_segtask:
                        loss = L * seg_loss + (1 - L) * cls_loss
                    else:
                        loss = cls_loss
                    cls_running_loss += cls_loss.item()

                    # 计算TP, FP, TN, FN
                    tp, fp, tn, fn = utils.GetTPFP(predicted, targets4)

                    epoch_tp += tp
                    epoch_fp += fp
                    epoch_tn += tn
                    epoch_fn += fn

                    if i == 0:
                        tmp_pre = predicted.detach().long().cpu().squeeze()
                        tmp_tar = targets4.cpu()

                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            precision, recall, f1_score, acc = \
                utils.PrintTrainInfo(_only_segtask, epoch, epoch_num, epoch_tp, epoch_fp, epoch_tn, epoch_fn, num_zero,
                                     num_one, tmp_pre, tmp_tar, writer, Iter)
            # 计时结束
            t.ticend()
            t.printtime(content)
            t.ticbegin()

            # 调整学习率
            lr_sch, optimizer = utils.AdjustLr(lr_sch, optimizer, epoch, lr_cos_epoch, lr_warm_epoch, num_epochs_decay,
                                               utils.GetCurrentLr(optimizer), lr_low, decay_step, decay_ratio)

            # 计算平均epoch_cls_loss
            epoch_cls_loss = utils.LossExport(cls_running_loss, seg_running_loss, running_loss, datas, writer, epoch,
                                              _have_segtask)

            # 保存模型策略
            utils.SaveModel(model, epoch, epoch_cls_loss, save_model_dir)

            # 输出分割指标
            if _have_segtask:
                print('train set segmentation output',
                      'SE = %.3f, PC = %.3f, F1 = %.3f, JS = %.3f, DC = %.3f, IOU = %.3f, Acc = %.3f' % (
                          sum(SElist) / len(SElist), sum(PClist) / len(PClist), sum(F1list) / len(F1list),
                          sum(JSlist) / len(JSlist),
                          sum(DClist) / len(DClist), sum(IOUlist) / len(IOUlist), sum(Acclist) / len(Acclist)))
                writer.add_scalars('train/IOU', {'IOU': sum(IOUlist) / len(IOUlist)}, epoch)
                writer.add_scalars('train/DC', {'DC': sum(DClist) / len(DClist)}, epoch)

            print('Iter = ', Iter)
            writer.add_scalars('Lr', {'lr': utils.GetCurrentLr(optimizer)}, epoch)
            if epoch % 3 == 0:
                # test.trainvalid('train', datas, model, device, writer, Iter, class_num, _have_segtask)
                test.trainvalid('valid', valid_loader, model, device, writer, Iter, class_num, _have_segtask,
                                _only_segtask)

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
        test.test('test', test_loader, model, device, class_num, _have_segtask, _only_segtask)
    print('\nFinished Testing\n')


def Train_Mnist():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.InitModel('Net', False)
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
            for i, data in enumerate(test_loader, 0):
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

    Train_breast('unetRcls_ocls2_0', 16, 300, 'unetr', 1e-3, False, False, False)
    Train_breast('unetRcls_ocls2_1', 20, 300, 'unetr', 1e-3, False, False, False)
    Train_breast('unetRcls_ocls2_3', 30, 300, 'unetr', 1e-3, False, False, False)
    # Train_breast('UNet_olseg_0', 10, 550, 'unet', 1e-4, False, True, True)
    # Train_breast('efficientnetb7_cls2_0', 30, 'efficientnet', 1e-4, True, False)
    # Train_breast('resnet101_cls2bce_1', 20, 'resnet101', 1e-5, True, False)
    # Train_breast('xception_cls2bce_1', 20, 'xception', 1e-5, True, False)
    # Train_breast('mobilenetv3_cls2bce_0', 40, 'mobilenetv3', 1e-6, True, False)
    # Train_breast('vgg16_bn_cls2_bce_1', 10, 500, 'vgg16_bn', 1e-6, True, False, False)

