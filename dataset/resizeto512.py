#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: resizeto512.py
@datatime: 7/21/2023 1:04 AM
"""
from torchvision import transforms as T
import csv
from PIL import Image
import os


def readCsv(csvfname):
    # read csv to list of lists
    print(csvfname)
    with open(csvfname, 'r', ) as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines


# 顶一个函数实现图像的resize，并保存新的csv文件
def resize_img(filepath_img_l, filepath_mask_l, filepath_contour_l, filepath_dist_l,
               savepath_img_l, savepath_mask_l, savepath_contour_l, savepath_dist_l,
               filepath_csv, savepath_csv, size, nodule_num, label,
               img_name):
    # 说明一下，图像是RGB通道的，mask是单通道的灰度图，contour是单通道的灰度图，distance是单通道的灰度图
    # 读取图像
    img = Image.open(filepath_img_l)
    # 读取mask
    mask = Image.open(filepath_mask_l)
    # 读取contour
    contour = Image.open(filepath_contour_l)
    # 读取distance
    dist = Image.open(filepath_dist_l)

    # resize图像
    img = img.resize(size)
    # resize mask
    mask = mask.resize(size)
    # resize contour
    contour = contour.resize(size)
    # resize dist
    dist = dist.resize(size)

    # 保存图像
    img.save(savepath_img_l + '/' + img_name)
    # 保存mask
    mask.save(savepath_mask_l + '/' + img_name)
    # 保存contour
    contour.save(savepath_contour_l + '/' + img_name)
    # 保存dist
    dist.save(savepath_dist_l + '/' + img_name)


    # 保存csv文件
    with open(savepath_csv, 'a') as f:
        f.write(img_name + ',' + label + ',' + nodule_num + '\n')


# 创建一个函数，实现对目标文件夹列表中的文件夹的检测，如果不存在就创建
def create_dir(dir_list):
    """
    :param dir_list: 目标文件夹列表
    :return: None
    """
    for dir in dir_list:
        if not os.path.exists(dir):
            os.mkdir(dir)


def main():
    filepath_img = '../class_out/clip_dataset/p_image'
    filepath_mask = '../class_out/clip_dataset/p_mask'
    filepath_contour = '../class_out/clip_dataset/p_contour'
    filepath_dist = '../class_out/clip_dataset/p_distance_D1'
    filepath_csv = '../class_out/train.csv'

    savepath_img = '../class_out/512/p_image_512'
    savepath_mask = '../class_out/512/p_mask_512'
    savepath_contour = '../class_out/512/p_contour_512'
    savepath_dist = '../class_out/512/p_distance_D1_512'
    savepath_csv = '../class_out/512/train_512.csv'
    pathlist = [savepath_img, savepath_mask, savepath_contour, savepath_dist]
    create_dir(pathlist)

    if os.path.exists(savepath_csv):
        # 如果文件存在，就删除再创建
        os.remove(savepath_csv)
        with open(savepath_csv, 'w') as f:
            f.write('ID' + ',' + 'CATE' + ',' + 'size' + '\n')
    else:
        with open(savepath_csv, 'w') as f:
            f.write('ID' + ',' + 'CATE' + ',' + 'size' + '\n')

    # 读取csv文件
    CTcsvlines = readCsv(filepath_csv)
    # 开始之前去除第一行的表头
    CTcsvlines.pop(0)

    # 遍历csv文件
    for line in CTcsvlines:
        # 获取img_name
        img_name = line[0]
        # 获取label
        label = line[1]
        # 获取nodule_num
        nodule_num = line[2]

        # 获取img的路径
        filepath_img_l = filepath_img + '/' + img_name
        # 获取mask的路径
        filepath_mask_l = filepath_mask + '/' + img_name
        # 获取contour的路径
        filepath_contour_l = filepath_contour + '/' + img_name
        # 获取distance的路径
        filepath_dist_l = filepath_dist + '/' + img_name

        # 调用resize_img函数
        resize_img(filepath_img_l, filepath_mask_l, filepath_contour_l, filepath_dist_l,
                   savepath_img, savepath_mask, savepath_contour, savepath_dist,
                   filepath_csv, savepath_csv, (512, 512), nodule_num, label,
                   img_name)

        print(img_name + ' is resized!')

if __name__ == '__main__':
    main()