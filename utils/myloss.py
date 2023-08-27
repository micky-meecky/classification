
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import utils


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


class SoftDiceLossold(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLossold, self).__init__()

    def forward(self, probs, targets):
        num = targets.size(0)
        smooth = 1

        # probs = F.sigmoid(logits)
        # m1 = probs.view(num, -1)
        # m2 = targets.view(num, -1)
        m1 = probs
        m2 = targets
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

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


class BCEWithLogitsLossCustomcls(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean', pos_weight=None):
        super(BCEWithLogitsLossCustomcls, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, input, target):

        # 计算logits
        logits = torch.sigmoid(input)

        # 计算交叉熵损失
        bce_loss = F.binary_cross_entropy_with_logits(input, target, weight=self.weight, reduction='none',
                                                      pos_weight=self.pos_weight)

        # 计算Focal Loss
        focal_loss = (1 - logits) ** self.gamma * target * bce_loss + (logits ** self.gamma) * (1 - target) * bce_loss

        if self.reduction == 'sum':
            # 计算总和
            loss = torch.sum(focal_loss)
        else:
            loss = torch.mean(focal_loss)

        return loss


class BCEWithLogitsLossCustom(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='sum', pos_weight=None):
        super(BCEWithLogitsLossCustom, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, input, target, log_vars):
        # 对log_vars进行限制
        log_vars = torch.clamp(log_vars, min=-0.)

        # 计算logits
        logits = torch.sigmoid(input)

        # 计算交叉熵损失
        bce_loss = F.binary_cross_entropy_with_logits(input, target, weight=self.weight, reduction='none',
                                                      pos_weight=self.pos_weight)

        # 计算Focal Loss
        focal_loss = (1 - logits) ** self.gamma * target * bce_loss + (logits ** self.gamma) * (1 - target) * bce_loss

        if self.reduction == 'sum':
            # 针对loss中的每一个元素，计算exp(-log_vars)
            precision2 = torch.exp(-log_vars)
            # 将loss中的每一个元素乘以precision2，并加上log_vars
            loss = focal_loss + (focal_loss * (0.5 * precision2) + 0.5 * log_vars)  # 乘以权重
            # 计算总和
            loss = torch.sum(loss)
        return loss

class SoftDiceLossNewvar(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLossNewvar, self).__init__()

    def forward(self, probs, targets, device, log_vars):
        num = targets.size(0)  # 获取batch_size
        smooth = 1
        # 初始化损失为0
        loss = 0
        # 对log_vars进行限制
        log_vars = torch.clamp(log_vars, min=-0.)
        precision1 = torch.exp(-log_vars)
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
            loss = loss + (loss * (0.5 * precision1) + 0.5 * log_vars)  # 乘以权重
        # loss = loss / num
        return loss


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

class MTLModel(torch.nn.Module):
    def __init__(self, n_hidden, n_output):
        super(MTLModel, self).__init__()

        self.net1 = nn.Sequential(nn.Linear(1, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output))
        self.net2 = nn.Sequential(nn.Linear(1, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output))

    def forward(self, x):
        return [self.net1(x), self.net2(x)]


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 编码器
        self.encoder1 = self.contracting_block(1, 64)
        self.encoder2 = self.contracting_block(64, 128)
        self.mlp = nn.Linear(128, 1)

        # 解码器
        self.decoder1 = self.expanding_block(128, 64)
        self.decoder2 = self.expanding_block(64, 1)
        self.outcov = nn.Conv2d(32, 1, kernel_size=1)


    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def expanding_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, in_channels // 2, kernel_size=2, stride=2)
        )
        return block

    def forward(self, x):
        # 编码器
        encoder1_output = self.encoder1(x)
        encoder2_output = self.encoder2(encoder1_output)
        # 分类器
        # 先对特征进行平均池化
        encoder2_output_cls = F.avg_pool2d(encoder2_output, kernel_size=encoder2_output.size()[2:])
        encoder2_output_cls = encoder2_output_cls.view(encoder2_output_cls.size(0), -1)
        cls = self.mlp(encoder2_output_cls)

        # 解码器
        decoder1_output = self.decoder1(encoder2_output)
        decoder2_output = self.decoder2(decoder1_output)
        # 出来通道还不是1，需要在进行一次卷积
        decoder2_output = self.outcov(decoder2_output)

        seg = torch.sigmoid(decoder2_output)
        return seg, cls


class CustomDataset(Dataset):
    def __init__(self, num_samples, image_size):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = np.random.rand(self.image_size, self.image_size, 1)
        mask = np.random.randint(0, 2, size=(self.image_size, self.image_size), dtype=np.uint8)

        image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float()
        mask_tensor = torch.from_numpy(mask).float()
        label_tensor = torch.tensor(np.random.randint(0, 2), dtype=torch.float32)

        return image_tensor, mask_tensor, label_tensor


if __name__ == "__main__":
    np.random.seed(0)

    # 创建虚拟数据集
    num_samples = 10  # 数据集样本数量
    image_size = 256  # 图像大小
    dataset = CustomDataset(num_samples, image_size)

    # 遍历数据集并打印示例数据
    for i in range(len(dataset)):
        image, mask, label = dataset[i]

        print(f"Sample {i + 1}:")
        print("Image shape:", image.shape)
        print("Mask shape:", mask.shape)
        print("Label:", label.item())
        print()

    # 设置训练参数
    num_epochs = 100
    batch_size = 4
    learning_rate = 0.01

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 创建模型
    model = UNet()
    # 创建损失函数
    criterion = SoftDiceLossNew()
    # 使用手写的 BCEWithLogitsLossCustom
    custom_loss = BCEWithLogitsLossCustom()

    # 定义设备，如果有GPU可以使用，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mtl = utils.MultiTaskLossWrapper(2, model, device)

    optimizer = torch.optim.Adam(mtl.parameters(), lr=0.001, eps=1e-07)
    # 训练模型
    total_step = len(dataloader)
    for epoch in range(num_epochs):
        for i, (images, masks, labels) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            # 前向传播
            seg_loss, cls_loss, loss, log_vars = mtl(images, masks, labels, criterion, custom_loss)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('log_vars', log_vars)

            # 打印训练信息
            if (i + 1) % 2 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}")

    print("Training finished!")
