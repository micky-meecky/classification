#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: contour_mask.py
@datatime: 11/6/2023 4:51 PM
"""
import os
from PIL import Image, ImageFilter, ImageChops

# 边缘图像和分割结果图像的文件夹路径
mask_folder = r'D:\projects\cls\classification\class_out\512\p_mask_512'
result_folder_template = r'D:\projects\cls\classification\SegImgSavePath\{}'

# 模型名称列表
models = [
    'SideSE2AgCBAMUNet_cls_seg_ch3_256_24_',
    'ResUnet_cls_seg_ch3_256_10',
    'AgUnet_cls_seg_ch3_256_10',
    'ANUnet_cls_seg_ch3_256_10',
    'Unet_cls_seg_ch3_256_10',
    'UNETR_cls_seg_ch3_256_10',
    'M_UNet_cls_seg_ch3_256_00_'
]

# 要处理的图片索引
edge_image_files = {file_name.split('.')[0]: file_name for file_name in os.listdir(mask_folder)}

# 对于每一个模型进行操作
for model in models:
    result_folder = result_folder_template.format(model)
    # 确保输出文件夹存在
    output_folder = os.path.join(result_folder, 'overlayed')
    os.makedirs(output_folder, exist_ok=True)

    # 获取该模型分割结果的所有图片文件名
    result_image_files = os.listdir(result_folder)

    # 对于每一个模型结果图像
    for result_image_file in result_image_files:
        image_idx = result_image_file.split('_')[0]  # 获取文件的序号部分
        if image_idx in edge_image_files:  # 确保mask文件夹中存在对应的文件
            mask_image_path = os.path.join(mask_folder, edge_image_files[image_idx])
            result_image_path = os.path.join(result_folder, result_image_file)

            # 读取mask图像
            mask_image = Image.open(mask_image_path).convert("L")  # 转换为灰度图像
            # 提取边缘
            edges = mask_image.filter(ImageFilter.FIND_EDGES)
            # 应用阈值处理以确保只有边缘被提取
            edges = edges.point(lambda x: 255 if x > 0 else 0)
            # 将边缘转换为红色，将非边缘部分设为透明
            red_edges = Image.new("RGBA", edges.size)
            for x in range(edges.width):
                for y in range(edges.height):
                    if edges.getpixel((x, y)) == 255:  # 边缘的像素
                        red_edges.putpixel((x, y), (255, 0, 0, 255))

            # 使用MaxFilter增加红色边缘的宽度
            red_edges = red_edges.filter(ImageFilter.MaxFilter(3))

            # 保存红色边缘图像进行检查
            # red_edges.save(os.path.join(output_folder, f"red_edge_{image_idx}.png"))

            # 读取分割结果图像，并确保它是不透明的
            result_image = Image.open(result_image_path).convert('RGBA')
            result_image.putalpha(255)

            # 确保结果图像也是 RGBA 模式
            result_image = result_image.convert('RGBA')

            # 调整红色边缘图像的大小以匹配分割结果图像的大小
            red_edges = red_edges.resize(result_image.size, Image.ANTIALIAS)

            # 使用 alpha_composite 叠加图像
            overlayed_image = Image.alpha_composite(result_image, red_edges)

            # 保存图像
            overlayed_image_path = os.path.join(output_folder, result_image_file)
            overlayed_image.save(overlayed_image_path)

            print(f"Overlayed image {result_image_file} saved.")
        else:
            print(f"No matching edge file for result image {result_image_file}.")


