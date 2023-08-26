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
import copy

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
from dataset.data_loader import get_loader_difficult, get_loader, DrawSavePic
from utils.tictoc import TicToc
import utils.evaluation as ue
from utils.myloss import SoftDiceLossNew, JaccardLoss, BCEWithLogitsLossCustom, SoftDiceLossNewvar, \
    BCEWithLogitsLossCustomcls
import test
from utils import utils
import multiprocessing as mp
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
    train_loader = DataLoader(train_dataset, batch_size=1024, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=1024, sampler=val_sampler)
    return train_loader, val_loader, test_loader


def getdataset(device, csv_file, fold_K, fold_idx, image_size, batch_size, testbs, num_workers, use_clip,
               validate_flag=True, channel=3, datasc='DDTI'):
    augmentation_prob = 0.5
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

    if use_clip:
        filepath_img = './class_out/clip_dataset/clip_image'
        filepath_mask = './class_out/clip_dataset/clip_mask'
        # filepath_contour = './class_out/clip_dataset/clip_contour'
        # filepath_dist = './class_out/clip_dataset/clip_distance_D1'
    else:
        if channel == 3:
            filepath_img = './class_out/512/p_image_512'
            filepath_mask = './class_out/512/p_mask_512'
            # filepath_contour = './class_out/512/p_contour'
            # filepath_dist = './class_out/512/p_distance_D1'
        else:
            # filepath_img = './class_out/stage1/p_image'
            # filepath_mask = './class_out/stage1/p_mask'
            if datasc == 'DDTI':
                filepath_img = './class_out/2_preprocessed_data/stage1/p_image'
                filepath_mask = './class_out/2_preprocessed_data/stage1/p_mask'
            if datasc == 'BUSI':
                filepath_img = './class_out/stage1/p_image'
                filepath_mask = './class_out/stage1/p_mask'
            # filepath_contour = './class_out/512/p_contour'
            # filepath_dist = './class_out/512/p_distance_D1'

    train_list = [filepath_img + sep + i[0] for i in train]
    train_list_GT = [filepath_mask + sep + i[0] for i in train]
    # trian_list_contour = [filepath_contour + sep + i[0] for i in train]
    # train_list_dist = [filepath_dist + sep + i[0] for i in train]
    train_class_list_GT = [i[1] for i in train]

    valid_list = [filepath_img + sep + i[0] for i in valid]
    valid_list_GT = [filepath_mask + sep + i[0] for i in valid]
    # valid_list_contour = [filepath_contour + sep + i[0] for i in valid]
    # valid_list_dist = [filepath_dist + sep + i[0] for i in valid]
    valid_class_list_GT = [i[1] for i in valid]

    test_list = [filepath_img + sep + i[0] for i in test]
    test_list_GT = [filepath_mask + sep + i[0] for i in test]
    # test_list_contour = [filepath_contour + sep + i[0] for i in test]
    # test_list_dist = [filepath_dist + sep + i[0] for i in test]
    test_class_list_GT = [i[1] for i in test]

    print("images count in train:{}".format(len(train_list)))
    print("images count in valid:{}".format(len(valid_list)))
    print("images count in test :{}".format(len(test_list)))

    # 将train_list, valid_list, test_list保存到一个txt文件中h
    train_list_txt = './foldinfo/train_list_' + str(fold_idx) + '.txt'
    valid_list_txt = './foldinfo/valid_list_' + str(fold_idx) + '.txt'
    test_list_txt = './foldinfo/test_list_' + str(fold_idx) + '.txt'
    utils.WriteIntoTxt(train_list, train_list_txt)
    utils.WriteIntoTxt(valid_list, valid_list_txt)
    utils.WriteIntoTxt(test_list, test_list_txt)

    # train_loader = get_loader_difficult(seg_list=None,
    train_loader = get_loader(seg_list=None,
                              GT_list=train_list_GT,
                              class_list=train_class_list_GT,
                              image_list=train_list,
                              # contour_list=trian_list_contour,
                              # dist_list=train_list_dist,
                              image_size=image_size,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              mode='train',
                              augmentation_prob=augmentation_prob,
                              device=device)

    # valid_loader = get_loader_difficult(seg_list=None,
    valid_loader = get_loader(seg_list=None,
                              GT_list=valid_list_GT,
                              class_list=valid_class_list_GT,
                              image_list=valid_list,
                              # contour_list=valid_list_contour,
                              # dist_list=valid_list_dist,
                              image_size=image_size,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              mode='val',
                              augmentation_prob=0.,
                              device=device)

    # test_loader = get_loader_difficult(seg_list=None,
    test_loader = get_loader(seg_list=None,
                             GT_list=test_list_GT,
                             class_list=test_class_list_GT,
                             image_list=test_list,
                             # contour_list=test_list_contour,
                             # dist_list=test_list_dist,
                             image_size=image_size,
                             batch_size=testbs,
                             num_workers=num_workers,
                             mode='test',
                             augmentation_prob=0.,
                             device=device)

    return train_loader, valid_loader, test_loader


def breast_loader(batch_size, testbs, device, validate_flag, use_clip, channel, size, datasc):
    if use_clip:
        csv_path = './class_out/clip_dataset/clip_train.csv'
    else:
        # train_path_m = './train_path/fold/fold'
        if datasc == 'DDTI':
            csv_path = './class_out/2_preprocessed_data/train.csv'
        if datasc == 'BUSI':
            csv_path = './class_out/train.csv'

    fold_k = 5
    fold_idx = 1
    # fold_id = 1
    # distance_type = "dist_mask"
    # normal_flag = False
    image_size = size
    num_workers = 2

    print('batch_size: ', batch_size)
    train_loader, valid_loader, test_loader = getdataset(device, csv_path, fold_k, fold_idx, image_size, batch_size,
                                                         testbs,
                                                         num_workers,
                                                         use_clip,
                                                         validate_flag,
                                                         channel, datasc)
    return train_loader, valid_loader, test_loader


def Train_breast(Project, Bs, epoch, Model_name, lr, Use_pretrained, _have_segtask, _only_segtask,
                 is_continue_train,
                 use_clip, channel, size, decayepoch, datasc, clsaux
                 ):
    project = Project  # project name-----------------------------------------------------
    epoch_num = epoch  # epoch_num -----------------------------------------------------
    class_num = 1  # class_num -----------------------------------------------------
    lr = lr  # 学习率  -----------------------------------------------------
    validate_flag = True  # 是否使用验证集 -----------------------------------------------------
    lr_low = 1e-15  # 学习率下限  ------------------------------------------------------
    lr_warm_epoch = 10  # warm up 的 epoch 数 -----------------------------------------------------
    lr_cos_epoch = decayepoch  # 学习率下降的epoch数 -----------------------------------------------------
    num_epochs_decay = 100  # 学习率下降的epoch数 -----------------------------------------------------
    decay_step = 10  # 学习率下降的epoch数 -----------------------------------------------------
    decay_ratio = 0.952  # 学习率下降的比例 -----------------------------------------------------
    bs = Bs  # batch_size -----------------------------------------------------
    testbs = 1  # test_batch_size -----------------------------------------------------
    L = 0.80  # 代表的是seg_loss的权重[现已作废，已有自适应调整策略] -----------------------------------------------------
    use_pretrained = Use_pretrained  # 是否使用预训练模型 -----------------------------------------------------
    model_name = Model_name  # 模型名字 ------------------------------------------------------
    log_dir = './log/log'
    model_dir = './savemodel'
    train_pic_list = './foldinfo/' + project + '/'
    SegImgSavePath = './SegImgSavePath/' + project
    save_model_dir = os.path.join(model_dir, project)
    # mp.set_start_method('spawn')
    t = TicToc()
    te = TicToc()
    content = "----per epoch Time: "
    contentvalid = "----per epoch training&vlidation test Time: "
    contentwholeepoch = "----whole epoch Time: "
    contenttotal = "----total cost: "
    is_train = True
    is_test = True  # False
    best_valid_acc = 0
    best_valid_score = 0
    is_continue_train = is_continue_train
    _only_segtask = _only_segtask
    if _only_segtask:
        _have_segtask = True
    else:
        _have_segtask = _have_segtask

    model = utils.InitModel(model_name, use_pretrained, class_num, _have_segtask,
                            _only_segtask, channel, size)  # ---------------------------------------------
    utils.init_weights(model)
    model, device = utils.Device(model)

    print(getModelSize(model))
    print('project: ', project)
    train_loader, valid_loader, test_loader = breast_loader(bs, testbs, device, validate_flag,
                                                            use_clip, channel, size, datasc)
    # train_loader, test_loader = OpenDataSet.SelectDataSet('Cifar_10', bs)
    if is_continue_train:
        model_dir = './savemodel/' + project + '/miniloss.pth'
        model.load_state_dict(torch.load(model_dir, map_location=device))
        print('load model')

    if _only_segtask:
        criterion_seg = SoftDiceLossNew()
        optimizer = optim.Adam(list(model.parameters()), lr, (0.5, 0.99))  # ----------------------------------------
    else:
        if use_clip:
            pos_weight = torch.tensor([3340 / 4344]).to(device)
        else:
            pos_weight = torch.tensor([500 / 122]).to(device)
        if _have_segtask:
            criterion_seg = SoftDiceLossNewvar()  # -----------------------------------------------------
            criterion_cls = BCEWithLogitsLossCustom(pos_weight=pos_weight)
            mtl = utils.MultiTaskLossWrapper(model, device)
            optimizer = optim.Adam(list(mtl.parameters()), lr,
                                   (0.5, 0.99))  # ----------------------------------------
        else:
            criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # criterion_cls = nn.NLLLoss()    # -----------------------------------------------------
            # criterion_seg = nn.BCELoss()  # -----------------------------------------------------
            optimizer = optim.Adam(list(model.parameters()), lr,
                                   (0.5, 0.99))  # ----------------------------------------

    lr_sch = utils.LrDecay(lr_warm_epoch, lr_cos_epoch, lr, lr_low, optimizer)  # -------------------------------

    utils.Mkdir(SegImgSavePath)
    utils.Mkdir(train_pic_list)

    if is_train:
        SElist = []
        PClist = []
        F1list = []
        JSlist = []
        DClist = []
        IOUlist = []
        Acclist = []
        torch.autograd.set_detect_anomaly(True)
        Iter = 0
        tmp_pre = tmp_tar = None
        utils.Mkdir(save_model_dir)
        log_dir = os.path.join(log_dir, project)
        utils.Mkdir(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        datas = train_loader  # -----------------------------------------------------
        utils.check_grad(model)
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
                (img_file_name, inputs, targets1, targets4) = data
                # 由于上面进行stack的时候必须保证相同大小的张量，从而targets1变成了三通道的，这里只需要第一个通道即可，维度保持微bs x 1 x 512 x 512
                # targets1 = targets1[:, 0, :, :].unsqueeze(1)
                if epoch == 0:
                    # 没必要每次试验，因为只需要案例图像就够了，不需要在每次实验的时候都保存。况且每一epoch的图都不一样
                    # DrawSavePic(img_file_name, inputs, targets1, targets2, targets3, train_pic_list)
                    pass
                optimizer.zero_grad()
                Iter += 1
                # (inputs, targets4) = data
                if class_num <= 2:
                    # 将标签进行修改, 0->0, 1->0, 2->1
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
                    # segout = torch.sigmoid(segout)
                    SR_flat = segout.view(segout.size(0), -1)
                    GT_flat = targets1.view(targets1.size(0), -1)
                    loss = criterion_seg(SR_flat, GT_flat, device)
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
                    else:
                        outputs = model(inputs)

                    if class_num > 2:
                        labels = F.softmax(outputs, dim=1)  # -----------------------------------------------------
                        _, predicted = torch.max(labels.data, 1)
                        # cls_loss = criterion_cls(outputs, targets4)
                    else:  # 如果是二分类，就用sigmoid
                        labels = torch.sigmoid(outputs)
                        predicted = torch.round(labels)  # 这是表示四舍五入，而不是取整数
                        targets4v = targets4.view(-1, 1)
                        targets4v = targets4v.to(torch.float)
                        # cls_loss = criterion_cls(outputs, targets4v)

                    if _have_segtask:
                        # 根据分类的结果，分类为1的代表的是没有结节的样本，而分类为0的代表的是有结节的样本。
                        # 没有结节的话，则将segout乘以0，有结节的话，则将segout乘以1
                        if clsaux:
                            cls_predicted = 1 - predicted  # 这里的predicted是0或者1，所以1-predicted就是1或者0
                            segout = segout * cls_predicted.view(-1, 1, 1, 1)  # 要转换成bs x 1 x 1 x 1
                        SR_flat = segout.view(segout.size(0), -1)
                        GT_flat = targets1.view(targets1.size(0), -1)
                        # seg_loss = criterion_seg(SR_flat, GT_flat, device)
                        # seg_running_loss += seg_loss.item()
                        SE, PC, F1, JS, DC, IOU, Acc = ue.get_all_seg(segout, targets1, device)
                        # 将这些指标存到一个list里面，方便后面计算平均值
                        SElist.append(SE)
                        PClist.append(PC)
                        F1list.append(F1)
                        JSlist.append(JS)
                        DClist.append(DC)
                        IOUlist.append(IOU)
                        Acclist.append(Acc)

                    if _have_segtask:
                        seg_loss, cls_loss, loss, log_vars = mtl(outputs, SR_flat, targets4v, GT_flat, criterion_seg,
                                                                 criterion_cls)
                        seg_running_loss += seg_loss.item()
                    else:
                        cls_loss = criterion_cls(outputs, targets4v)
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
            # print('log_vars:', log_vars.data.tolist())
            # log_vars: [0.02311073988676071, -0.027175873517990112], 将这个里的每个元素分别求个exp再输出
            # exp_log_vars = [torch.exp(-log_var) for log_var in log_vars]
            # print('exp_log_vars:', exp_log_vars)

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
            epoch_cls_loss, epoch_loss = utils.LossExport(cls_running_loss, seg_running_loss, running_loss, datas,
                                                          writer, epoch,
                                                          _have_segtask)

            # 保存模型策略
            utils.SaveModel(model, epoch, epoch_loss, save_model_dir)

            # 输出分割指标
            if _have_segtask:
                print('train set segmentation output',
                      'SE = %.3f, PC = %.3f, F1 = %.3f, JS = %.3f, DC = %.3f, IOU = %.3f, Acc = %.3f' % (
                          sum(SElist) / len(SElist), sum(PClist) / len(PClist), sum(F1list) / len(F1list),
                          sum(JSlist) / len(JSlist),
                          sum(DClist) / len(DClist), sum(IOUlist) / len(IOUlist), sum(Acclist) / len(Acclist)))
                writer.add_scalars('train/IOU', {'IOU': sum(IOUlist) / len(IOUlist)}, epoch)
                writer.add_scalars('train/DC', {'DC': sum(DClist) / len(DClist)}, epoch)
                # writer.add_scalars('train/log_vars0', {'log_vars0': log_vars[0]}, epoch)
                # writer.add_scalars('train/log_vars1', {'log_vars1': log_vars[1]}, epoch)

            print('Iter = ', Iter)
            writer.add_scalars('Lr', {'lr': utils.GetCurrentLr(optimizer)}, epoch)
            if epoch % 3 == 0:
                if _have_segtask:
                    if _only_segtask:
                        valid_iou = test.trainvalid('valid', valid_loader, model, device, writer, Iter,
                                                    class_num,
                                                    _have_segtask,
                                                    _only_segtask)
                        valid_score = valid_iou
                    else:
                        valid_acc, valid_iou = test.trainvalid('valid', valid_loader, model, device, writer, Iter,
                                                               class_num,
                                                               _have_segtask,
                                                               _only_segtask)
                        valid_score = valid_acc + valid_iou
                    if valid_score > best_valid_score:
                        best_valid_score = valid_score
                        best_epoch = epoch
                        best_model_wts = copy.deepcopy(model.state_dict())
                        print('best_valid_score = ', best_valid_score)
                        print('best_epoch = ', best_epoch)
                        # 保存模型
                        best_model = save_model_dir + '/best' + '.pth'
                        torch.save(best_model_wts, best_model)
                        print('best model saved at epoch %d' % epoch)
                else:
                    valid_acc = test.trainvalid('valid', valid_loader, model, device, writer, Iter, class_num,
                                                _have_segtask,
                                                _only_segtask)
                    if valid_acc > best_valid_acc:
                        best_valid_acc = valid_acc
                        best_epoch = epoch
                        print('best_valid_acc = ', best_valid_acc)
                        print('best_epoch = ', best_epoch)
                        best_model = save_model_dir + '/best' + '.pth'
                        torch.save(model.state_dict(), best_model)
                        print('best model saved at epoch %d' % epoch)

            t.ticend()
            t.printtime(contentvalid)
            te.ticend()
            te.printtime(contentwholeepoch)
            te.printlefttime(epoch, epoch_num)
        t.printtime(contenttotal, True)
        writer.close()
        # torch.save(model.state_dict(), 'model.pth ')

    print('Finished Training\n')
    if is_test:
        #测试最后一epoch 的模型效果并输出
        test_precision, test_recall, test_f1_score, test_acc = \
            test.test('test', test_loader, model, SegImgSavePath, device, class_num, _have_segtask, _only_segtask)
        print('test_precision, test_recall, test_f1_score, test_acc:', test_precision, test_recall, test_f1_score,
              test_acc)
        print('最后一epoch的模型效果测试完毕')
        mini_loss_model = save_model_dir + '/best' + '.pth'
        model.load_state_dict(torch.load(mini_loss_model, map_location=device))
        test_precision, test_recall, test_f1_score, test_acc = \
            test.test('test', test_loader, model, SegImgSavePath, device, class_num, _have_segtask, _only_segtask)
    print('\nFinished Testing\n')
    # test(model)

    return test_precision, test_recall, test_f1_score, test_acc


def Train_Mnist():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.InitModel('swin-vit', False, class_num=3, _have_segtask=False, _only_segtask=False)
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
    optimizer = optim.AdamW(model.parameters(), lr=0.0002)

    is_train = True
    is_test = True
    is_continue_train = False

    if is_continue_train:
        model.load_state_dict(torch.load('model.pth'))
        print('load model')

    if is_train:
        for epoch in range(100):
            correct = 0
            total = 0
            running_loss = 0.0
            for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                # 对outputs进行激活，然后计算损失，使用softmax激活
                outputs = F.softmax(outputs, dim=1)
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

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 50000 train images: %.4f %%' % (100 * correct / total))

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


def main():
    # Train_Mnist()
    # Train_breast('unetRcls_ocls2_5', 30, 200, 'unetr', 1e-4, False, False, False, is_continue_train=False)
    # Train_breast('UNet_olseg_0', 10, 600, 'unet', 1e-2, False, True, True, False)
    # Train_breast('unetRseg_cls_seg_8', 5, 100, 'unetr', 9.63366620781354e-14, False, True, _only_segtask=False,
    #              is_continue_train=True)

    base_name1 = 'Unet_ocls_3'
    base_name2 = 'Unet_ocls_4'
    # 从A到Z
    name_order = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z']

    lr_list = (1e-1, 4e-2, 1e-2, 8e-3, 6e-3, 4e-3, 2e-3, 1e-3, 8e-4, 6e-4,
               4e-4, 2e-4, 1e-4, 8e-5, 6e-5, 4e-5, 2e-5, 1e-5, 8e-6, 6e-6,
               4e-6, 2e-6, 1e-6)  # lr_list是学习率元组。目前设置了23个学习率

    testp = []
    testr = []
    testf1 = []
    testacc = []

    # 尝试不同的学习率，分两个批次，一次是奇数，一次是偶数，奇数的使用仅含有z12的，偶数的使用仅含有cls_token的。
    for j in range(len(lr_list)):
        test_precision, test_recall, test_f1_score, test_acc = \
            Train_breast(base_name2 + name_order[j], 6, 400, 'unet', lr_list[j],
                         Use_pretrained=False,
                         _have_segtask=False,
                         _only_segtask=False,
                         is_continue_train=False,
                         use_clip=False)
        testp.append(test_precision)
        testr.append(test_recall)
        testf1.append(test_f1_score)
        testacc.append(test_acc)

        for i in range(len(testp)):
            print('第' + str(i + 1) + '个实验结果：', end=', ')
            print(testp[i], end=', ')
            print(testr[i], end=', ')
            print(testf1[i], end=', ')
            print(testacc[i])

    print(testp)
    print(testr)
    print(testf1)
    print(testacc)
    print('end')


if __name__ == '__main__':
    # main(    )
    testp = []
    testr = []
    testf1 = []
    testacc = []

    # test_precision, test_recall, test_f1_score, test_acc = \
    #     Train_breast('SideSEUNet_cls_seg_ch3_256_03', 6, 800, 'SideSEUNet', 6e-4,
    #                  Use_pretrained=False,
    #                  _have_segtask=True,
    #                  _only_segtask=False,
    #                  is_continue_train=False,
    #                  use_clip=False,
    #                  channel=3,
    #                  size=256,
    #                  decayepoch=790,
    #                  datasc='BUSI',
    #                  clsaux=False)
    # testp.append(test_precision)
    # testr.append(test_recall)
    # testf1.append(test_f1_score)
    # testacc.append(test_acc)
    #
    # for i in range(len(testp)):
    #     print('第' + str(i + 1) + '个实验结果：', end=', ')
    #     print(testp[i], end=', ')
    #     print(testr[i], end=', ')
    #     print(testf1[i], end=', ')
    #     print(testacc[i])

    test_precision, test_recall, test_f1_score, test_acc = \
        Train_breast('SideSEAgCBAMUNet_cls_seg_ch3_256_00', 6, 800, 'SideAgCBAMUNet', 6e-4,
                     Use_pretrained=False,
                     _have_segtask=True,
                     _only_segtask=False,
                     is_continue_train=False,
                     use_clip=False,
                     channel=3,
                     size=256,
                     decayepoch=790,
                     datasc='BUSI',
                     clsaux=False)
    testp.append(test_precision)
    testr.append(test_recall)
    testf1.append(test_f1_score)
    testacc.append(test_acc)

    for i in range(len(testp)):
        print('第' + str(i + 1) + '个实验结果：', end=', ')
        print(testp[i], end=', ')
        print(testr[i], end=', ')
        print(testf1[i], end=', ')
        print(testacc[i])

    # test_precision, test_recall, test_f1_score, test_acc = \
    #     Train_breast('CBAMUNet_cls_seg_ch3_256_00', 6, 800, 'CBAMUNet', 6e-4,
    #                  Use_pretrained=False,
    #                  _have_segtask=True,
    #                  _only_segtask=False,
    #                  is_continue_train=False,
    #                  use_clip=False,
    #                  channel=3,
    #                  size=256,
    #                  decayepoch=790,
    #                  datasc='BUSI',
    #                  clsaux=False)
    # testp.append(test_precision)
    # testr.append(test_recall)
    # testf1.append(test_f1_score)
    # testacc.append(test_acc)
    #
    # for i in range(len(testp)):
    #     print('第' + str(i + 1) + '个实验结果：', end=', ')
    #     print(testp[i], end=', ')
    #     print(testr[i], end=', ')
    #     print(testf1[i], end=', ')
    #     print(testacc[i])


    '''
    test_precision, test_recall, test_f1_score, test_acc = \
        Train_breast('preres101CBAMAGUnet_cls_seg_ch3_512_01', 6, 800, 'res101UNet', 6e-4,
                     Use_pretrained=True,
                     _have_segtask=True,
                     _only_segtask=False,
                     is_continue_train=False,
                     use_clip=False,
                     channel=3,
                     size=512,
                     decayepoch=790,
                     datasc='BUSI')
    testp.append(test_precision)
    testr.append(test_recall)
    testf1.append(test_f1_score)
    testacc.append(test_acc)

    for i in range(len(testp)):
        print('第' + str(i + 1) + '个实验结果：', end=', ')
        print(testp[i], end=', ')
        print(testr[i], end=', ')
        print(testf1[i], end=', ')
        print(testacc[i])

    test_precision, test_recall, test_f1_score, test_acc = \
        Train_breast('InDilatedUNet_cls_seg_ch3_512_00', 6, 800, 'InDilatedUNet', 6e-4,
                     Use_pretrained=False,
                     _have_segtask=True,
                     _only_segtask=False,
                     is_continue_train=False,
                     use_clip=False,
                     channel=3,
                     size=256,
                     decayepoch=790,
                     datasc='BUSI')
    testp.append(test_precision)
    testr.append(test_recall)
    testf1.append(test_f1_score)
    testacc.append(test_acc)

    for i in range(len(testp)):
        print('第' + str(i + 1) + '个实验结果：', end=', ')
        print(testp[i], end=', ')
        print(testr[i], end=', ')
        print(testf1[i], end=', ')
        print(testacc[i])


    test_precision, test_recall, test_f1_score, test_acc = \
        Train_breast('ResUNet_cls_seg_ch3_256_01', 6, 800, 'ResUNet', 6e-4,
                     Use_pretrained=False,
                     _have_segtask=True,
                     _only_segtask=False,
                     is_continue_train=False,
                     use_clip=False,
                     channel=3,
                     size=256,
                     decayepoch=790,
                     datasc='BUSI')
    testp.append(test_precision)
    testr.append(test_recall)
    testf1.append(test_f1_score)
    testacc.append(test_acc)

    for i in range(len(testp)):
        print('第' + str(i + 1) + '个实验结果：', end=', ')
        print(testp[i], end=', ')
        print(testr[i], end=', ')
        print(testf1[i], end=', ')
        print(testacc[i])

    test_precision, test_recall, test_f1_score, test_acc = \
        Train_breast('AgCBAMPixViTUNet_cls_seg_ch3_256_01', 6, 800, 'AgCBAMPixViTUNet', 6e-4,
                     Use_pretrained=False,
                     _have_segtask=True,
                     _only_segtask=False,
                     is_continue_train=False,
                     use_clip=False,
                     channel=3,
                     size=256,
                     decayepoch=790,
                     datasc='BUSI')
    testp.append(test_precision)
    testr.append(test_recall)
    testf1.append(test_f1_score)
    testacc.append(test_acc)

    for i in range(len(testp)):
        print('第' + str(i + 1) + '个实验结果：', end=', ')
        print(testp[i], end=', ')
        print(testr[i], end=', ')
        print(testf1[i], end=', ')
        print(testacc[i])

    test_precision, test_recall, test_f1_score, test_acc = \
        Train_breast('AgCBAMUNet_cls_seg_ch3_512_01', 6, 800, 'AgCBAMUNet', 6e-4,
                     Use_pretrained=False,
                     _have_segtask=True,
                     _only_segtask=False,
                     is_continue_train=False,
                     use_clip=False,
                     channel=3,
                     size=512,
                     decayepoch=790,
                     datasc='BUSI')
    testp.append(test_precision)
    testr.append(test_recall)
    testf1.append(test_f1_score)
    testacc.append(test_acc)

    for i in range(len(testp)):
        print('第' + str(i + 1) + '个实验结果：', end=', ')
        print(testp[i], end=', ')
        print(testr[i], end=', ')
        print(testf1[i], end=', ')
        print(testacc[i])



'''

