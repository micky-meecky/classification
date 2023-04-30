#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: test.py
@datatime: 4/21/2023 10:24 AM
"""

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import utils.evaluation as ue
import torch.nn.functional as F

def trainvalid(mode: str, dataloader: DataLoader, model, device: torch.device, writer, Iter):
    printcontent = mode + 'set testing...'
    print(printcontent)
    outputcontent = mode + ' set上的准确率: %.3f %%'
    scalarcontent = mode + ' accuracy'
    segoutputcontent = mode + ' segmentation output'

    # 训练集上测试
    i = 0
    SElist = []
    PClist = []
    F1list = []
    JSlist = []
    DClist = []
    IOUlist = []
    Acclist = []
    cls_acclist = []

    with torch.no_grad():
        for data in dataloader:
            (img_file_name, images, targets1, targets2, targets3, targets4) = data
            if torch.cuda.is_available():
                images = images.to(device)
                targets1 = targets1.to(device)
                targets4 = targets4.to(device)
            SR, labels = model(images)
            # SR = F.sigmoid(segout)
            # outputs = F.softmax(outputs, dim=1)     # -----------------------------------------------------
            labels = torch.exp(labels)  # -----------------------------------------------------
            _, predicted = torch.max(labels.data, 1)
            SE, PC, F1, JS, DC, IOU, Acc = ue.get_all_seg(SR, targets1)
            cls_acc = ue.get_clsaccuracy(predicted, targets4)
            # 将这些指标存到一个list里面，方便后面计算平均值
            SElist.append(SE)
            PClist.append(PC)
            F1list.append(F1)
            JSlist.append(JS)
            DClist.append(DC)
            IOUlist.append(IOU)
            Acclist.append(Acc)
            cls_acclist.append(cls_acc)

            # 输出第一批的预测结果
            if i == 0:
                predicted = predicted.cpu()
                targets4 = targets4.cpu()
                print('predicted = ', predicted)
                print('targets4 = ', targets4)
                i += 1

            # total += targets4.size(0)
            # correct += (predicted == targets4).sum().item()
        print(outputcontent % (100 * sum(cls_acclist) / len(cls_acclist)))
        # 输出seg的指标
        print(segoutputcontent, 'SE = %.3f, PC = %.3f, F1 = %.3f, JS = %.3f, DC = %.3f, IOU = %.3f, Acc = %.3f' % (
            sum(SElist) / len(SElist), sum(PClist) / len(PClist), sum(F1list) / len(F1list), sum(JSlist) / len(JSlist),
            sum(DClist) / len(DClist), sum(IOUlist) / len(IOUlist), sum(Acclist) / len(Acclist)))

    writer.add_scalars('Accuracy', {scalarcontent: (100 * sum(cls_acclist) / len(cls_acclist))}, Iter)


def test(mode: str, dataloader: DataLoader, model, device: torch.device):
    printcontent = mode + 'set testing...'
    print(printcontent)
    outputcontent = mode + ' set上的准确率: %.3f %%'
    segoutputcontent = mode + ' segmentation output'

    # 训练集上测试
    i = 0
    SElist = []
    PClist = []
    F1list = []
    JSlist = []
    DClist = []
    IOUlist = []
    Acclist = []
    cls_acclist = []

    with torch.no_grad():
        for data in dataloader:
            (img_file_name, images, targets1, targets2, targets3, targets4) = data
            if torch.cuda.is_available():
                images = images.to(device)
                targets1 = targets1.to(device)
                targets4 = targets4.to(device)
            SR, labels = model(images)
            # SR = F.sigmoid(segout)
            # outputs = F.softmax(outputs, dim=1)     # -----------------------------------------------------
            labels = torch.exp(labels)  # -----------------------------------------------------
            _, predicted = torch.max(labels.data, 1)
            SE, PC, F1, JS, DC, IOU, Acc = ue.get_all_seg(SR, targets1)
            cls_acc = ue.get_clsaccuracy(predicted, targets4)
            # 将这些指标存到一个list里面，方便后面计算平均值
            SElist.append(SE)
            PClist.append(PC)
            F1list.append(F1)
            JSlist.append(JS)
            DClist.append(DC)
            IOUlist.append(IOU)
            Acclist.append(Acc)
            cls_acclist.append(cls_acc)

            # 输出第一批的预测结果
            if i == 0:
                predicted = predicted.cpu()
                targets4 = targets4.cpu()
                print('predicted = ', predicted)
                print('targets4 = ', targets4)
                i += 1

            # total += targets4.size(0)
            # correct += (predicted == targets4).sum().item()
        print(outputcontent % (100 * sum(cls_acclist) / len(cls_acclist)))
        # 输出seg的指标
        print(segoutputcontent, 'SE = %.3f, PC = %.3f, F1 = %.3f, JS = %.3f, DC = %.3f, IOU = %.3f, Acc = %.3f\n' % (
            sum(SElist) / len(SElist), sum(PClist) / len(PClist), sum(F1list) / len(F1list), sum(JSlist) / len(JSlist),
            sum(DClist) / len(DClist), sum(IOUlist) / len(IOUlist), sum(Acclist) / len(Acclist)))
