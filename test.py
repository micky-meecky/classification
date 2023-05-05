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
from torch.utils.data import DataLoader
import utils.evaluation as ue
import torch.nn.functional as F


def trainvalid(mode: str, dataloader: DataLoader, model,
               device: torch.device, writer, Iter,
               class_num, _have_segtask: bool):
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

    with torch.no_grad():
        epoch_tp, epoch_fp, epoch_tn, epoch_fn = 0, 0, 0, 0
        for data in dataloader:
            (img_file_name, images, targets1, targets2, targets3, targets4) = data
            # (images, targets4) = data
            if class_num == 2:
                # 将标签进行修改
                targets4[targets4 == 0] = 0
                targets4[targets4 == 1] = 0
                targets4[targets4 == 2] = 1
            if torch.cuda.is_available():
                images = images.to(device)
                # targets1 = targets1.to(device)
                targets4 = targets4.to(device)
            if _have_segtask:
                SR, labels = model(images)  # -----------------------------------------------------
                # SR = F.sigmoid(segout)
                # SE, PC, F1, JS, DC, IOU, Acc = ue.get_all_seg(SR, targets1)
                # 将这些指标存到一个list里面，方便后面计算平均值
                # SElist.append(SE)
                # PClist.append(PC)
                # F1list.append(F1)
                # JSlist.append(JS)
                # DClist.append(DC)
                # IOUlist.append(IOU)
                # Acclist.append(Acc)
            else:
                labels = model(images)
                # labels = torch.exp(labels)  # -----------------------------------------------------
                if class_num > 2:
                    labels = F.softmax(labels, dim=1)  # -----------------------------------------------------
                    _, predicted = torch.max(labels.data, 1)
                else:  # 如果是二分类，就用sigmoid
                    labels = torch.sigmoid(labels)
                    predicted = torch.round(labels.data)

                # 计算TP, FP, TN, FN
                tp = torch.sum((predicted == 0) & (targets4 == 0)).item()
                fp = torch.sum((predicted == 0) & (targets4 == 1)).item()
                tn = torch.sum((predicted == 1) & (targets4 == 1)).item()
                fn = torch.sum((predicted == 1) & (targets4 == 0)).item()

                epoch_tp += tp
                epoch_fp += fp
                epoch_tn += tn
                epoch_fn += fn

            # 输出第一批的预测结果, 以及最后一批的预测结果
            if i == 0 or i == len(dataloader) - 1:
                predicted = predicted.detach()
                predicted = predicted.long()
                predicted = predicted.cpu()
                targets4 = targets4.cpu()
                predicted = predicted.squeeze()
                predicted = predicted.cpu()
                targets4 = targets4.cpu()
                print('predicted = ', predicted)
                print('targets4 = ', targets4)
                i += 1

            # total += targets4.size(0)
            # correct += (predicted == targets4).sum().item()

        # 计算精确率（Precision）、召回率（Recall）和F1分数
        precision = epoch_tp / (epoch_tp + epoch_fp) if epoch_tp + epoch_fp > 0 else 0
        recall = epoch_tp / (epoch_tp + epoch_fn) if epoch_tp + epoch_fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        acc = (epoch_tp + epoch_tn) / (epoch_tp + epoch_tn + epoch_fp + epoch_fn)
        print(
            f' Precision: {precision:.4f},'
            f' Recall: {recall:.4f},'
            f' F1-score: {f1_score:.4f},'
            f' Accuracy: {acc:.4f}'
        )
        # 输出seg的指标
        if _have_segtask:
            print(segoutputcontent, 'SE = %.3f, PC = %.3f, F1 = %.3f, JS = %.3f, DC = %.3f, IOU = %.3f, Acc = %.3f' % (
                sum(SElist) / len(SElist), sum(PClist) / len(PClist), sum(F1list) / len(F1list), sum(JSlist) / len(JSlist),
                sum(DClist) / len(DClist), sum(IOUlist) / len(IOUlist), sum(Acclist) / len(Acclist)))

    # writer.add_scalars('Accuracy', {scalarcontent: (100 * sum(cls_acclist) / len(cls_acclist))}, Iter)
    writer.add_scalars('Accuracy', {'valid acc': acc}, Iter)
    writer.add_scalars('precision', {'valid precision': precision}, Iter)
    writer.add_scalars('recall', {'valid recall': recall}, Iter)
    writer.add_scalars('f1_score', {'valid f1_score': f1_score}, Iter)


def test(mode: str, dataloader: DataLoader, model, device: torch.device, class_num, _have_segtask: bool):
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
        epoch_tp, epoch_fp, epoch_tn, epoch_fn = 0, 0, 0, 0
        for data in dataloader:
            (img_file_name, images, targets1, targets2, targets3, targets4) = data
            # (images, targets4) = data
            if class_num == 2:
                # 将标签进行修改
                targets4[targets4 == 0] = 0
                targets4[targets4 == 1] = 0
                targets4[targets4 == 2] = 1
            if torch.cuda.is_available():
                images = images.to(device)
                # targets1 = targets1.to(device)
                targets4 = targets4.to(device)
            if _have_segtask:
                SR, labels = model(images)  # -----------------------------------------------------
                # SR = F.sigmoid(segout)
                # SE, PC, F1, JS, DC, IOU, Acc = ue.get_all_seg(SR, targets1)
                # 将这些指标存到一个list里面，方便后面计算平均值
                # SElist.append(SE)
                # PClist.append(PC)
                # F1list.append(F1)
                # JSlist.append(JS)
                # DClist.append(DC)
                # IOUlist.append(IOU)
                # Acclist.append(Acc)
            else:
                labels = model(images)
                # labels = torch.exp(labels)  # -----------------------------------------------------
                if class_num > 2:
                    labels = F.softmax(labels, dim=1)  # -----------------------------------------------------
                else:  # 如果是二分类，就用sigmoid
                    labels = torch.sigmoid(labels)
                _, predicted = torch.max(labels.data, 1)
                # cls_acc = ue.get_clsaccuracy(predicted, targets4)
                # cls_acclist.append(cls_acc)

                # 计算TP, FP, TN, FN
                tp = torch.sum((predicted == 0) & (targets4 == 0)).item()
                fp = torch.sum((predicted == 0) & (targets4 == 1)).item()
                tn = torch.sum((predicted == 1) & (targets4 == 1)).item()
                fn = torch.sum((predicted == 1) & (targets4 == 0)).item()

                epoch_tp += tp
                epoch_fp += fp
                epoch_tn += tn
                epoch_fn += fn

            # 输出预测结果
            if i % 10 == 0:     # 每10个batch输出一次
                predicted = predicted.cpu()
                targets4 = targets4.cpu()
                print('predicted = ', predicted)
                print('targets4 = ', targets4)
                i += 1


        # 计算精确率（Precision）、召回率（Recall）和F1分数
        precision = epoch_tp / (epoch_tp + epoch_fp) if epoch_tp + epoch_fp > 0 else 0
        recall = epoch_tp / (epoch_tp + epoch_fn) if epoch_tp + epoch_fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        acc = (epoch_tp + epoch_tn) / (epoch_tp + epoch_tn + epoch_fp + epoch_fn)
        print(
            f' Precision: {precision:.4f},'
            f' Recall: {recall:.4f},'
            f' F1-score: {f1_score:.4f},'
            f' Accuracy: {acc:.4f}'
        )
        # print(outputcontent % (100 * sum(cls_acclist) / len(cls_acclist)))
        if _have_segtask:
            # 输出seg的指标
            print(segoutputcontent, 'SE = %.3f, PC = %.3f, F1 = %.3f, JS = %.3f, DC = %.3f, IOU = %.3f, Acc = %.3f\n' % (
                sum(SElist) / len(SElist), sum(PClist) / len(PClist), sum(F1list) / len(F1list), sum(JSlist) / len(JSlist),
                sum(DClist) / len(DClist), sum(IOUlist) / len(IOUlist), sum(Acclist) / len(Acclist)))


