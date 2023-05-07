
import torch
import torch.nn as nn
import numpy as np


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        num = targets.size(0)   # batch_size
        if num == 1:
            print('Error: SoftDiceLoss dice loss can not be computed with batch_size = 1')
        if num == 0:
            print('Error: SoftDiceLoss dice loss can not be computed with batch_size = 0')
        smooth = 1

        # 针对每一个batch中的每一个样本，计算其dice loss
        loss = 0
        for i in range(num):
            m1 = probs[i]
            m2 = targets[i]
            # 先将m1, m2类型从tensor转换为numpy
            SR = m1.data.cpu().numpy()
            GT = m2.data.cpu().numpy()
            # 将m1, m2中的值转换为0或1
            SR = (SR > 0.5).astype(np.float64)
            GT = (GT == np.max(GT)).astype(np.float64)
            intersection = (m1 * m2)
            # 计算acc
            corr = np.sum(SR == GT)
            acc = float(corr) / float(SR.shape[0])
            score = 1.  # 转换为tensor
            score = torch.from_numpy(np.array(score)).float().cuda()
            if acc != 1:
                score = 2. * (intersection.sum() + smooth) / (m1.sum() + m2.sum() + smooth)
            loss += 1 - score
        loss = loss / num
        return loss

class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(JaccardLoss, self).__init__()

    def forward(self, probs, targets):
        num = targets.size(0)
        smooth = 1

        loss = 0
        for i in range(num):
            m1 = probs[i]
            m2 = targets[i]
            # 先将m1, m2类型从tensor转换为numpy
            SR = m1.data.cpu().numpy()
            GT = m2.data.cpu().numpy()
            # 将m1, m2中的值转换为0或1
            SR = (SR > 0.5).astype(np.float64)
            GT = (GT == np.max(GT)).astype(np.float64)
            intersection = (m1 * m2)
            # 计算acc
            corr = np.sum(SR == GT)
            acc = float(corr) / float(SR.shape[0])
            if acc == 1:
                score = 1
            else:
                score = (intersection.sum() + smooth) / (m1.sum() + m2.sum() - intersection.sum() + smooth)
            loss += 1 - score

