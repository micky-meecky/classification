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
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import utils.evaluation as ue
import torch.nn.functional as F


def trainvalid(mode: str, dataloader: DataLoader, model,
               device: torch.device, writer, Iter,
               class_num, _have_segtask: bool, _only_segtask: bool,
               deepsup: bool, clsaux: bool):
    printcontent = mode + 'set testing...'
    print(printcontent)
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
            (img_file_name, images, targets1, targets4) = data
            targets1 = targets1[:, 0, :, :].unsqueeze(1)
            # (images, targets4) = data
            if class_num <= 2:
                # 将标签进行修改
                targets4[targets4 == 0] = 0
                targets4[targets4 == 1] = 0
                targets4[targets4 == 2] = 1
            if torch.cuda.is_available():
                images = images.to(device)
                # targets1 = targets1.to(device)
                targets4 = targets4.to(device)
            if _only_segtask:
                if deepsup is False:
                    targets1 = targets1.to(device)
                    segout = model(images)
                    segout = torch.sigmoid(segout)
                    SE, PC, F1, JS, DC, IOU, Acc = ue.get_all_seg(segout, targets1, device)
                    # 将这些指标存到一个list里面，方便后面计算平均值
                    SElist.append(SE)
                    PClist.append(PC)
                    F1list.append(F1)
                    JSlist.append(JS)
                    DClist.append(DC)
                    IOUlist.append(IOU)
                    Acclist.append(Acc)
                else:
                    targets1 = targets1.to(device)
                    _, _, _, segout0_4 = model(images)
                    segout0_4 = torch.sigmoid(segout0_4)
                    SE, PC, F1, JS, DC, IOU, Acc = ue.get_all_seg(segout0_4, targets1, device)
                    SElist.append(SE)
                    PClist.append(PC)
                    F1list.append(F1)
                    JSlist.append(JS)
                    DClist.append(DC)
                    IOUlist.append(IOU)
                    Acclist.append(Acc)
            else:
                if _have_segtask:
                    if deepsup is False:
                        labels, segout = model(images)
                        segout = torch.sigmoid(segout)
                    else:
                        labels, _, _, _, segout0_4 = model(images)
                        segout0_4 = torch.sigmoid(segout0_4)
                else:
                    labels = model(images)
                    # labels = torch.exp(labels)  # -----------------------------------------------------
                if class_num > 2:
                    labels = F.softmax(labels, dim=1)  # -----------------------------------------------------
                    _, predicted = torch.max(labels.data, 1)
                else:  # 如果是二分类，就用sigmoid
                    labels = torch.sigmoid(labels)
                    predicted = torch.round(labels.data)

                if _have_segtask:
                    if deepsup is False:
                        if clsaux:
                            cls_predicted = 1 - predicted  # 这里的predicted是0或者1，所以1-predicted就是1或者0
                            segout = segout * cls_predicted.view(-1, 1, 1, 1)  # 要转换成bs x 1 x 1 x 1
                        SE, PC, F1, JS, DC, IOU, Acc = ue.get_all_seg(segout, targets1, device)
                        # 将这些指标存到一个list里面，方便后面计算平均值
                        SElist.append(SE)
                        PClist.append(PC)
                        F1list.append(F1)
                        JSlist.append(JS)
                        DClist.append(DC)
                        IOUlist.append(IOU)
                        Acclist.append(Acc)
                    else:
                        if clsaux:
                            cls_predicted = 1 - predicted
                            segout0_4 = segout0_4 * cls_predicted.view(-1, 1, 1, 1)
                        SE, PC, F1, JS, DC, IOU, Acc = ue.get_all_seg(segout0_4, targets1, device)
                        SElist.append(SE)
                        PClist.append(PC)
                        F1list.append(F1)
                        JSlist.append(JS)
                        DClist.append(DC)
                        IOUlist.append(IOU)
                        Acclist.append(Acc)

                # 计算TP, FP, TN, FN
                predicted = predicted.squeeze().long()
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
                    predicted = predicted.detach().cpu().long().squeeze()
                    targets4 = targets4.cpu()
                    print('predicted = ', predicted)
                    print('targets4 = ', targets4)
                    i += 1

        print('epoch_tp = ', epoch_tp, 'epoch_fp = ', epoch_fp, 'epoch_tn = ', epoch_tn, 'epoch_fn = ', epoch_fn)
        if not _only_segtask:
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
            # writer.add_scalars('Accuracy', {scalarcontent: (100 * sum(cls_acclist) / len(cls_acclist))}, Iter)
            writer.add_scalars('Accuracy', {'valid acc': acc}, Iter)
            writer.add_scalars('precision', {'valid precision': precision}, Iter)
            writer.add_scalars('recall', {'valid recall': recall}, Iter)
            writer.add_scalars('f1_score', {'valid f1_score': f1_score}, Iter)
        # 输出seg的指标
        if _have_segtask:
            if not _only_segtask:
                print(segoutputcontent, 'SE = %.3f, PC = %.3f, F1 = %.3f, JS = %.3f, DC = %.3f, IOU = %.3f, Acc = %.3f' % (
                    sum(SElist) / len(SElist), sum(PClist) / len(PClist), sum(F1list) / len(F1list),
                    sum(JSlist) / len(JSlist),
                    sum(DClist) / len(DClist), sum(IOUlist) / len(IOUlist), sum(Acclist) / len(Acclist)))
                writer.add_scalars('valid/IOU', {'IOU': sum(IOUlist) / len(IOUlist)}, Iter)
                writer.add_scalars('valid/DC', {'DC': sum(DClist) / len(DClist)}, Iter)
                iou = sum(IOUlist) / len(IOUlist)
                return acc, iou
            else:
                print(segoutputcontent, 'SE = %.3f, PC = %.3f, F1 = %.3f, JS = %.3f, DC = %.3f, IOU = %.3f, Acc = %.3f' % (
                    sum(SElist) / len(SElist), sum(PClist) / len(PClist), sum(F1list) / len(F1list),
                    sum(JSlist) / len(JSlist),
                    sum(DClist) / len(DClist), sum(IOUlist) / len(IOUlist), sum(Acclist) / len(Acclist)))
                writer.add_scalars('valid/IOU', {'IOU': sum(IOUlist) / len(IOUlist)}, Iter)
                writer.add_scalars('valid/DC', {'DC': sum(DClist) / len(DClist)}, Iter)
                iou = sum(IOUlist) / len(IOUlist)
                return iou
        else:
            return acc


def test(mode: str, dataloader: DataLoader, model, SegImgSavePath, device: torch.device,
         class_num, _have_segtask: bool, _only_segtask: bool, deepsup: bool, clsaux: bool):
    printcontent = mode + 'set testing...'
    print(printcontent)
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
            (img_file_name, images, targets1, targets4) = data
            targets1 = targets1[:, 0, :, :].unsqueeze(1)
            # (images, targets4) = data
            if class_num <= 2:
                # 将标签进行修改
                targets4[targets4 == 0] = 0
                targets4[targets4 == 1] = 0
                targets4[targets4 == 2] = 1
            if torch.cuda.is_available():
                images = images.to(device)
                # targets1 = targets1.to(device)
                targets4 = targets4.to(device)
            if _only_segtask:
                if deepsup is False:
                    targets1 = targets1.to(device)
                    segout = model(images)
                    segout = torch.sigmoid(segout)
                    SE, PC, F1, JS, DC, IOU, Acc = ue.get_all_seg(segout, targets1, device)
                    # 将这些指标存到一个list里面，方便后面计算平均值
                    SElist.append(SE)
                    PClist.append(PC)
                    F1list.append(F1)
                    JSlist.append(JS)
                    DClist.append(DC)
                    IOUlist.append(IOU)
                    Acclist.append(Acc)
                else:
                    targets1 = targets1.to(device)
                    _, _, _, segout0_4 = model(images)
                    segout0_4 = torch.sigmoid(segout0_4)
                    SE, PC, F1, JS, DC, IOU, Acc = ue.get_all_seg(segout0_4, targets1, device)
                    SElist.append(SE)
                    PClist.append(PC)
                    F1list.append(F1)
                    JSlist.append(JS)
                    DClist.append(DC)
                    IOUlist.append(IOU)
                    Acclist.append(Acc)
            else:
                if _have_segtask:
                    if deepsup is False:
                        labels, segout = model(images)
                        segout = torch.sigmoid(segout)
                    else:
                        labels, _, _, _, segout0_4 = model(images)
                        segout0_4 = torch.sigmoid(segout0_4)
                else:
                    labels = model(images)
                    # labels = torch.exp(labels)  # -----------------------------------------------------
                if class_num > 2:
                    labels = F.softmax(labels, dim=1)  # -----------------------------------------------------
                    _, predicted = torch.max(labels.data, 1)
                else:  # 如果是二分类，就用sigmoid
                    labels = torch.sigmoid(labels)
                    predicted = torch.round(labels.data)

                if _have_segtask:
                    if deepsup is False:
                        if clsaux:
                            cls_predicted = 1 - predicted  # 这里的predicted是0或者1，所以1-predicted就是1或者0
                            segout = segout * cls_predicted.view(-1, 1, 1, 1)  # 要转换成bs x 1 x 1 x 1
                        SE, PC, F1, JS, DC, IOU, Acc = ue.get_all_seg(segout, targets1, device)
                        # 将这些指标存到一个list里面，方便后面计算平均值
                        SElist.append(SE)
                        PClist.append(PC)
                        F1list.append(F1)
                        JSlist.append(JS)
                        DClist.append(DC)
                        IOUlist.append(IOU)
                        Acclist.append(Acc)
                    else:
                        if clsaux:
                            cls_predicted = 1 - predicted
                            segout0_4 = segout0_4 * cls_predicted.view(-1, 1, 1, 1)
                        SE, PC, F1, JS, DC, IOU, Acc = ue.get_all_seg(segout0_4, targets1, device)
                        SElist.append(SE)
                        PClist.append(PC)
                        F1list.append(F1)
                        JSlist.append(JS)
                        DClist.append(DC)
                        IOUlist.append(IOU)
                        Acclist.append(Acc)

                # 计算TP, FP, TN, FN
                predicted = predicted.squeeze().long()
                tp = torch.sum((predicted == 1) & (targets4 == 1)).item()
                fp = torch.sum((predicted == 1) & (targets4 == 0)).item()
                tn = torch.sum((predicted == 0) & (targets4 == 0)).item()
                fn = torch.sum((predicted == 0) & (targets4 == 1)).item()

                epoch_tp += tp
                epoch_fp += fp
                epoch_tn += tn
                epoch_fn += fn

                # 输出第一批的预测结果, 以及最后一批的预测结果
                if i == 0 or i == len(dataloader) - 1:
                    predicted = predicted.detach().cpu().long().squeeze()
                    targets4 = targets4.cpu()
                    print('predicted = ', predicted)
                    print('targets4 = ', targets4)
                    i += 1

                # 输出并保存分割图像
                if _have_segtask:
                    pass
                    # segout = segout.squeeze()
                    # segout = segout.cpu()
                    # segout = segout.detach().numpy()
                    # segout = np.round(segout)
                    # segout = segout.astype(np.uint8)
                    # segout = segout * 255
                    # segout = Image.fromarray(segout)
                    # # 获取图片名字, ./class_out/stage1/p_image\\86.png
                    # img_file_name = img_file_name[0].split('\\')[1].split('.')  # ['86', 'png']
                    # img_file_name = img_file_name[0]  # '86'
                    # # 将分类的结果predicted转为str
                    # predicted = str(predicted.item())
                    # # 将predicted添加到图片名字后面没比如86_0.png
                    # img_file_name = img_file_name + '_' + predicted
                    # segout.save(SegImgSavePath + '/' + img_file_name + '.png')

        print('epoch_tp = ', epoch_tp, 'epoch_fp = ', epoch_fp, 'epoch_tn = ', epoch_tn, 'epoch_fn = ', epoch_fn)
        if not _only_segtask:
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
        else:
            precision = 0
            recall = 0
            f1_score = 0
            acc = 0
        # 输出seg的指标
        if _have_segtask:
            print(segoutputcontent, 'SE = %.3f, PC = %.3f, F1 = %.3f, JS = %.3f, DC = %.3f, IOU = %.3f, Acc = %.3f' % (
                sum(SElist) / len(SElist), sum(PClist) / len(PClist), sum(F1list) / len(F1list),
                sum(JSlist) / len(JSlist),
                sum(DClist) / len(DClist), sum(IOUlist) / len(IOUlist), sum(Acclist) / len(Acclist)))

    return precision, recall, f1_score, acc
