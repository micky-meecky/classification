#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: GradCam.py
@datatime: 11/6/2023 10:32 AM
"""
import os

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from mymodels.Unet import UNet, UNet, UNetcls, UNetseg, Res101UNet, AgUNet, AgUNetseg, ResUNet, InDilatedUNet, SideUNet, \
    CasDilatedUNet, M_UNet_seg, SideDiUNet
from mymodels.UnetPP import UNetPlusPlusSeg, DSUNetPlusPlusSeg, DSUNetPlusPlus
from mymodels.unetr import UNETR
from mymodels.CBAMUnet import SideAgCBAMUNet
import matplotlib.pyplot as plt


# 假设我们有一个简化的Unet模型，只有编码器和解码器各两层
class SimpleUnet(nn.Module):
    def __init__(self):
        super(SimpleUnet, self).__init__()
        # 编码器
        self.enc_conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 解码器
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        # 分类器
        self.classifier = nn.Linear(128 * 64 * 64, 2) # 假设输入图像大小为256x256

    def forward(self, x):
        # 编码
        x1 = F.relu(self.enc_conv1(x))
        x2 = F.relu(self.enc_conv2(x1))
        # 解码
        x3 = F.relu(self.dec_conv1(x2))
        x4 = self.dec_conv2(x3)
        # 分类
        logits = self.classifier(x2.view(x2.size(0), -1))
        return logits, x4


def math():
    origin_height = 100  # 原始高度
    decay_rate = 2  # 衰减率
    bounce_num = 10  # 弹跳次数
    Total_height = 0  # 总共弹跳高度
    Final_height = 0  # 最终弹跳高度
    middle_height = origin_height  # 中间弹跳高度
    Total_height = origin_height
    for i in range(bounce_num):
        middle_height /= decay_rate
        Total_height += middle_height * 2
        print("第%d次弹跳高度为%f米" % (i + 1, middle_height))

    # 在循环结束后计算最后一次上升的高度
    Final_height = middle_height
    # 在总距离中减去最后一次下落的高度（因为最后一次反弹不再下落）
    Total_height -= (middle_height * 2)

    print("第10次落地时，球共经过%f米" % Total_height)
    print("第10次反弹高度为%f米" % Final_height)


# 用于获取激活图的函数
def grad_cam(model, x, target_class):
    # 正向传播
    model.eval()
    x.requires_grad_()
    class_output, _ = model(x)

    # 反向传播
    model.zero_grad()
    class_loss = class_output[:, target_class].sum()
    class_loss.backward(retain_graph=True)

    # 获取梯度和特征
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # 应用梯度到特征上
    activations = model.get_activations().detach()
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]

    # 生成激活图
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    return heatmap.data.cpu().numpy()


def grad_cam_unetr(model, x, target_category):

    model.eval()

    x.requires_grad_()
    _, cls_out = model(x)
    prediction = torch.sigmoid(cls_out)
    model.zero_grad()
    if target_category == 1:
        target = prediction.sum()
    else:
        target = (1 - prediction).sum()
    target.backward()
    gradients = model.get_activations_gradient()
    if gradients is None:
        raise ValueError("No gradients found. Ensure that forward hook is set properly.")

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations()
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().detach().numpy()

    return heatmap


def grad_cam_binary(model, x, target_category):
    """
    Generate Grad-CAM for a binary classification task.

    Parameters:
    model: The trained model.
    x: The input image tensor.
    target_category: The target category (usually 1 or 0).

    Returns:
    The generated heatmap.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Forward pass to get model output
    x.requires_grad = True
    model_output, _ = model(x)

    # Apply sigmoid to get the prediction probability
    prediction = torch.sigmoid(model_output)

    # Calculate gradients for the prediction probability
    model.zero_grad()
    if target_category == 1:
        class_loss = prediction.sum()
    else:
        class_loss = (1 - prediction).sum()
    class_loss.backward(retain_graph=True)

    # Extract gradients
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Get model activations
    activations = model.get_activations().detach()

    # Multiply weights with gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Generate heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    # Convert to numpy array for visualization
    heatmap = heatmap.cpu().numpy()

    return heatmap


def grad_cam_binary_seg(model, x, target_category):
    """
    Generate Grad-CAM for a binary classification task.

    Parameters:
    model: The trained model.
    x: The input image tensor.
    target_category: The target category (usually 1 or 0).

    Returns:
    The generated heatmap.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Forward pass to get model output
    x.requires_grad = True
    _, model_output = model(x)

    # Apply sigmoid to get the prediction probability
    prediction = torch.sigmoid(model_output)

    # Calculate gradients for the prediction probability
    model.zero_grad()
    if target_category == 1:
        class_loss = prediction.sum()
    else:
        class_loss = (1 - prediction).sum()
    class_loss.backward(retain_graph=True)

    # Extract gradients
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Get model activations
    activations = model.get_activations().detach()

    # Multiply weights with gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Generate heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    # Convert to numpy array for visualization
    heatmap = heatmap.cpu().numpy()

    return heatmap


# 图片预处理
def process_image(RGB_G, image_path, target_size):
    image = Image.open(image_path)
    # 转为灰度图
    image = image.convert(RGB_G)
    # 调整图片大小
    image = image.resize(target_size, Image.LANCZOS)
    # 转换为张量
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 根据你的情况可能需要调整这些值
        #                      std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # 添加一个批次维度
    return image


def load_model(device, model_name, model_path):
    # 定义模型
    if model_name == 'UNETR':
        model = UNETR()
    elif model_name == 'AgUnet':
        model = AgUNet(3, 1)
    elif model_name == 'Unet':
        model = UNet(3, 1)
    elif model_name == 'M_UNet':
        model = M_UNet_seg(3, 1)
    elif model_name == 'ANUnet':
        model = UNetPlusPlusSeg(3, 1)
    elif model_name == 'ResUnet':
        model = ResUNet(3, 1)
    elif model_name == 'SideSE2AgCBAMUNet':
        model = SideAgCBAMUNet(3, 1)
    else:
        raise ValueError('model_name error')

    mini_loss_model = model_path
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(mini_loss_model))
    else:
        # 如果只有 CPU，将所有的模型权重映射到 CPU
        state_dict = torch.load(mini_loss_model, map_location=torch.device('cpu'))
        # 创建一个新的状态字典，其中所有的键名都没有 'module.' 前缀
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    model.to(device)
    return model


def GradCam():
    # 生成一个随机图像
    # img = np.random.randn(1, 3, 256, 256)
    # img = torch.from_numpy(img).float()
    models = [
        'SideSE2AgCBAMUNet_cls_seg_ch3_256_24_',
        'ResUnet_cls_seg_ch3_256_10',
        'AgUnet_cls_seg_ch3_256_10',
        'ANUnet_cls_seg_ch3_256_10',
        'Unet_cls_seg_ch3_256_10',
        'UNETR_cls_seg_ch3_256_10',
        'M_UNet_cls_seg_ch3_256_00_'
    ]
    models_name = {
        'SideSE2AgCBAMUNet_cls_seg_ch3_256_24_':
            'SideSE2AgCBAMUNet',
        'ResUnet_cls_seg_ch3_256_10':
            'ResUnet',
        'AgUnet_cls_seg_ch3_256_10':
            'AgUnet',
        'ANUnet_cls_seg_ch3_256_10':
            'ANUnet',
        'Unet_cls_seg_ch3_256_10':
            'Unet',
        'UNETR_cls_seg_ch3_256_10':
            'UNETR',
        'M_UNet_cls_seg_ch3_256_00_':
            'M_UNet'
    }
    model_paths = {
        'SideSE2AgCBAMUNet_cls_seg_ch3_256_24_':
            'D:/projects/cls/classification/savemodel/SideSE2AgCBAMUNet_cls_seg_ch3_256_24/best.pth',
        'ResUnet_cls_seg_ch3_256_10':
            'D:/projects/cls/classification/savemodel/ResUnet_cls_seg_ch3_256_10/best.pth',
        'AgUnet_cls_seg_ch3_256_10':
            'D:/projects/cls/classification/savemodel/AgUnet_cls_seg_ch3_256_10/best.pth',
        'ANUnet_cls_seg_ch3_256_10':
            'D:/projects/cls/classification/savemodel/ANUnet_cls_seg_ch3_256_10/best.pth',
        'Unet_cls_seg_ch3_256_10':
            'D:/projects/cls/classification/savemodel/Unet_cls_seg_ch3_256_10/best.pth',
        'UNETR_cls_seg_ch3_256_10':
            'D:/projects/cls/classification/savemodel/UNETR_cls_seg_ch3_256_10/best.pth',
        'M_UNet_cls_seg_ch3_256_00_':
            'D:/projects/cls/classification/savemodel/M_UNet_cls_seg_ch3_256_00/best.pth'
    }

    test_folder = 'D:/projects/cls/classification/SegImgSavePath/Unet_cls_seg_ch3_256_10'
    original_folder = 'D:/projects/cls/classification/class_out/512/p_image_512'
    gradcam_save_folder = 'D:/projects/cls/classification/SegImgSavePath/GradCAM_4_seg'

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in models:
        path_name = model_name
        if model_name in models_name.keys():
            model_name = models_name[model_name]
        if model_name != 'SideSE2AgCBAMUNet':
            continue
        # 加载模型
        model = load_model(device, model_name, model_paths[path_name])
        model.eval()

        # 创建保存 Grad-CAM 的目录
        save_path = os.path.join(gradcam_save_folder, model_name)
        os.makedirs(save_path, exist_ok=True)

        # 遍历测试集
        for test_image_name in os.listdir(test_folder):
            if test_image_name.endswith('.png') and not os.path.isdir(os.path.join(test_folder, test_image_name)):
                # 获取原始图像的编号
                original_image_num = test_image_name.split('_')[0]
                original_image_path = os.path.join(original_folder, original_image_num + '.png')

                # 检查原始图像是否存在
                if os.path.exists(original_image_path):
                    # 加载并处理图片
                    if model_name == 'UNETR':
                        image = process_image('L', original_image_path, (224, 224)).to(device)
                        # 获取Grad-CAM
                        heatmap = grad_cam_unetr(model, image, 0)
                    else:
                        image = process_image('RGB', original_image_path, (256, 256)).to(device)
                        # 获取Grad-CAM
                        heatmap = grad_cam_binary_seg(model, image, 0)

                    # 保存 Grad-CAM
                    heatmap = cv2.resize(heatmap, (256, 256))
                    # 假设 heatmap 是你计算出的激活图
                    heatmap = np.clip(heatmap, 0, 1)  # 确保值在0和1之间
                    heatmap = np.nan_to_num(heatmap)  # 将 NaN 和 Inf 替换为0

                    # 现在转换为uint8应该不会有问题了
                    heatmap = np.uint8(255 * heatmap)

                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                    cv2.imwrite(os.path.join(save_path, test_image_name), heatmap)

                    print('Grad-CAM of {} saved.'.format(test_image_name))
                else:
                    print('Original image of {} not exists.'.format(test_image_name))
            else:
                print('{} is not a image.'.format(test_image_name))

    print('Grad-CAM saved.')

if __name__ == '__main__':
    GradCam()

    # math()

