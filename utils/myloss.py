
import torch
import torch.nn as nn
import numpy as np


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        num = targets.size(0)  # 获取batch_size
        if num == 0:
            print('Error: SoftDiceLoss dice loss can not be computed with batch_size = 0')
        smooth = 1

        # 初始化损失为0
        loss = 0
        for i in range(num):
            m1 = probs[i].clone()  # 创建新的tensor以避免修改原tensor
            m2 = targets[i].clone()  # 创建新的tensor以避免修改原tensor

            # 直接在PyTorch的tensor上进行操作，避免转化为numpy
            SR = (m1 > 0.5).float()
            if torch.any(m2 > 0):
                GT = (m2 == torch.max(m2)).float()
            else:
                GT = torch.zeros_like(m2)
            intersection = (m1 * m2)

            # 计算acc
            corr = torch.sum(SR == GT)
            acc = float(corr) / float(SR.shape[0])
            score = torch.tensor(1., requires_grad=True).to(device)  # 创建需要求导的tensor
            if acc != 1:
                m1sum = m1.sum()
                m2sum = m2.sum()
                intersum = intersection.sum()
                score = (2. * intersum + smooth) / (m1sum + m2sum + smooth)
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
            if np.any(GT > 0):
                GT = (GT == np.max(GT)).astype(np.float64)
            else:
                GT = np.zeros_like(GT, dtype=np.float64)
            intersection = (m1 * m2)
            # 计算acc
            corr = np.sum(SR == GT)
            acc = float(corr) / float(SR.shape[0])
            if acc == 1:
                score = 1
            else:
                score = (intersection.sum() + smooth) / (m1.sum() + m2.sum() - intersection.sum() + smooth)
            loss += 1 - score


class SoftDiceLossNew(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLossNew, self).__init__()

    def forward(self, probs, targets, device):
        num = targets.size(0)  # 获取batch_size
        smooth = 1
        # 初始化损失为0
        loss = 0
        for i in range(num):
            m1 = probs[i]
            m2 = targets[i]

            # 直接在PyTorch的tensor上进行操作，避免转化为numpy
            SR = (m1 > 0.5).float()
            if torch.any(m2 > 0):
                GT = (m2 == torch.max(m2)).float()
            else:
                GT = torch.zeros_like(m2)
            intersection = (m1 * m2)

            # 计算acc
            corr = torch.sum(SR == GT)
            acc = float(corr) / float(SR.shape[0])
            if acc != 1:
                m1sum = m1.sum()
                m2sum = m2.sum()
                intersum = intersection.sum()
                score = (2. * intersum + smooth) / (m1sum + m2sum + smooth)
            else:
                score = torch.tensor(1., requires_grad=True).to(device)  # 创建需要求导的tensor
            loss = loss + 1 - score
        loss = loss / num
        return loss


if __name__ == "__main__":
    # 创建损失函数
    criterion = SoftDiceLossNew()

    # 定义设备，如果有GPU可以使用，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建一些模拟数据,probs是预测的概率图像，创建一个全0的概率图像
    probs = torch.zeros((10, 1, 256, 256)).float().to(device)  # 10个1通道256x256的概率图像
    targets = torch.ones((10, 1, 256, 256)).float().to(device)  # 10个1通道256x256的标签图像

    # 除了第一维，flatten predictions and targets
    probs = probs.view(probs.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    # 计算损失
    loss = criterion(probs, targets, device)

    # 反向传播一下测试
    loss.backward()

    # 输出损失
    print("Loss:", loss.item())
