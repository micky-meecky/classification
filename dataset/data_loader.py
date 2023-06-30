import os
import random

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
import warnings
from utils import utils
import matplotlib.pyplot as plt
import numpy as np
from dataset import img_mask_aug

warnings.filterwarnings('ignore',
                        message='Argument \'interpolation\' of type int is deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.')
# from torchvision.transforms import InterpolationMode
from torchvision import models
import torch.nn as nn
import torch.optim as optim
# import sleep的库
import time

# 设置随机种子
# random_seed = 2023
# random.seed(random_seed)
sep = os.sep  # os.sep根据你所处的平台，自动采用相应的分隔符号

class ImageFolder_new(data.Dataset):
    def __init__(self, seg_list, GT_list, class_list, contour_list, dist_list, image_list, image_size=512,
                 mode='train', augmentation_prob=0.4, load_preseg=False, device='cpu'):
        """Initializes image paths and preprocessing module."""
        self.GT_paths = GT_list
        self.image_paths = image_list
        self.class_list = class_list
        self.contour_paths = contour_list
        self.dist_paths = dist_list

        self.load_preseg = load_preseg
        if self.load_preseg:
            self.seg_paths = seg_list

        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270, 45, 135, 215, 305]
        # self.RotationDegree = [0,90,180,270,45]
        self.augmentation_prob = augmentation_prob
        self.resize_range = [224, 224]
        self.CropRange = [200, 224]  # 注意,上界貌似不能大于resize的下界,待验证
        self.device = device

    # print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        GT_path = self.GT_paths[index]
        class_item = self.class_list[index]
        # 将class_item转化为tensor
        class_item = torch.tensor(int(class_item))

        filename = image_path.split('/')[-1]
        # GT_path = os.path.join(self.GT_paths, filename)

        image_o = image = Image.open(image_path)
        GT_o = GT = Image.open(GT_path)
        contour_o = contour = Image.open(self.contour_paths[index])
        dist_o = dist = Image.open(self.dist_paths[index])
        # image = image.convert('RGB')

        if self.load_preseg:
            seg_path = self.seg_paths[index]
            seg_o = seg = Image.open(seg_path)

        aspect_ratio = image.size[1] / image.size[0]

        # 先将PIL转为tensor
        transform = T.Compose([T.ToTensor()])
        transform_GT = T.Compose([T.ToTensor()])
        transform_contour = T.Compose([T.ToTensor()])
        transform_dist = T.Compose([T.ToTensor()])

        image = transform(image)
        GT = transform_GT(GT)
        contour = transform_contour(contour)
        dist = transform_dist(dist)

        # 将images这些放在device，即GPU上
        image = image.to(self.device)
        GT = GT.to(self.device)
        contour = contour.to(self.device)
        dist = dist.to(self.device)

        Transform = []
        Transform_GT = []  # 注意,GT的插值需要最近邻nearest,但是采取非线性插值可能有奇效
        Transform_contour = []
        Transform_dist = []

        ResizeRange = random.randint(self.resize_range[0], self.resize_range[1])

        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:

            # 修改亮度、对比度。
            Transform.append(T.ColorJitter(brightness=0.25, contrast=0.25))
            Transform_GT.append(T.ColorJitter(brightness=0.25, contrast=0.25))
            Transform_contour.append(T.ColorJitter(brightness=0.25, contrast=0.25))
            Transform_dist.append(T.ColorJitter(brightness=0.25, contrast=0.25))

            T.Grayscale(num_output_channels=3),  # 随机转为灰度图

            # Transform.append(T.RandomApply([T.RandomErasing(p=0.5)], p=0.5))
            # Transform_GT.append(T.RandomApply([T.RandomErasing(p=0.5)], p=0.5))
            # Transform_contour.append(T.RandomApply([T.RandomErasing(p=0.5)], p=0.5))
            # Transform_dist.append(T.RandomApply([T.RandomErasing(p=0.5)], p=0.5))

            # Transform.append(
            #     T.Resize((int(ResizeRange * aspect_ratio), ResizeRange), interpolation=Image.BICUBIC))  # 双三次
            # Transform_GT.append(
            #     T.Resize((int(ResizeRange * aspect_ratio), ResizeRange), interpolation=Image.NEAREST))  # 最近邻
            # Transform_contour.append(
            #     T.Resize((int(ResizeRange * aspect_ratio), ResizeRange), interpolation=Image.NEAREST))  # 最近邻
            # Transform_dist.append(
            #     T.Resize((int(ResizeRange * aspect_ratio), ResizeRange), interpolation=Image.NEAREST))  # 最近邻

            RotationDegree = random.randint(0, 7)
            RotationDegree = self.RotationDegree[RotationDegree]
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1 / aspect_ratio  # 这里交换一下宽高比,因为旋转了

            Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))  # RandomRotation的两个参数分别是旋转角度的下界和上界
            Transform_GT.append(T.RandomRotation((RotationDegree, RotationDegree)))
            Transform_contour.append(T.RandomRotation((RotationDegree, RotationDegree)))
            Transform_dist.append(T.RandomRotation((RotationDegree, RotationDegree)))

            # 在大旋转间隔基础上,微小调整旋转角度
            RotationRange = random.randint(-10, 10)
            Transform.append(T.RandomRotation((RotationRange, RotationRange)))
            Transform_GT.append(T.RandomRotation((RotationRange, RotationRange)))
            Transform_contour.append(T.RandomRotation((RotationRange, RotationRange)))
            Transform_dist.append(T.RandomRotation((RotationRange, RotationRange)))

            CropRange = random.randint(self.CropRange[0], self.CropRange[1])
            Transform.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
            Transform_GT.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
            Transform_contour.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
            Transform_dist.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))

            Transform = T.Compose(Transform)
            Transform_GT = T.Compose(Transform_GT)
            Transform_contour = T.Compose(Transform_contour)
            Transform_dist = T.Compose(Transform_dist)

            image = Transform(image)  # 对image进行transform操作
            GT = Transform_GT(GT)
            contour = Transform_contour(contour)
            dist = Transform_dist(dist)
            if self.load_preseg:
                seg = Transform_GT(seg)

            # crop
            # ShiftRange_left = random.randint(0, 20)
            # ShiftRange_upper = random.randint(0, 20)
            # ShiftRange_right = image.size[0] - random.randint(0, 20)
            # ShiftRange_lower = image.size[1] - random.randint(0, 20)
            # image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
            # GT = GT.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
            # contour = contour.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
            # dist = dist.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
            # if self.load_preseg:
            #     seg = seg.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))

            # crop
            # ShiftRange_left = random.randint(0, 20)
            # ShiftRange_upper = random.randint(0, 20)
            # ShiftRange_right = image.size(2) - random.randint(0, 20)
            # ShiftRange_lower = image.size(1) - random.randint(0, 20)
            #
            # image = image[:, ShiftRange_upper:ShiftRange_lower, ShiftRange_left:ShiftRange_right]
            # GT = GT[:, ShiftRange_upper:ShiftRange_lower, ShiftRange_left:ShiftRange_right]
            # contour = contour[:, ShiftRange_upper:ShiftRange_lower, ShiftRange_left:ShiftRange_right]
            # dist = dist[:, ShiftRange_upper:ShiftRange_lower, ShiftRange_left:ShiftRange_right]

            if self.load_preseg:
                seg = seg[:, ShiftRange_upper:ShiftRange_lower, ShiftRange_left:ShiftRange_right]

            # # flip
            # if random.random() < 0.5:
            #     image = F.hflip(image)
            #     GT = F.hflip(GT)
            #     contour = F.hflip(contour)
            #     dist = F.hflip(dist)
            #     if self.load_preseg:
            #         seg = F.hflip(seg)
            # if random.random() < 0.5:
            #     image = F.vflip(image)
            #     GT = F.vflip(GT)
            #     contour = F.vflip(contour)
            #     dist = F.vflip(dist)
            #     if self.load_preseg:
            #         seg = F.vflip(seg)

            # flip
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])
                GT = torch.flip(GT, dims=[2])
                contour = torch.flip(contour, dims=[2])
                dist = torch.flip(dist, dims=[2])
                if self.load_preseg:
                    seg = torch.flip(seg, dims=[2])

            if random.random() < 0.5:
                image = torch.flip(image, dims=[1])
                GT = torch.flip(GT, dims=[1])
                contour = torch.flip(contour, dims=[1])
                dist = torch.flip(dist, dims=[1])
                if self.load_preseg:
                    seg = torch.flip(seg, dims=[1])

            # Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)
            # image = Transform(image)

            Transform = []
            Transform_GT = []
            Transform_contour = []
            Transform_dist = []

        # if want to check iamge%GT while debug
        # plt.subplot(2,2,1)
        # plt.imshow(np.array(image_o),cmap=plt.cm.gray)
        # plt.subplot(2,2,2)
        # plt.imshow(np.array(GT_o),cmap=plt.cm.gray)
        # plt.subplot(2,2,3)
        # plt.imshow(np.array(image),cmap=plt.cm.gray)
        # plt.subplot(2,2,4)
        # plt.imshow(np.array(GT),cmap=plt.cm.gray)
        # plt.show()

        final_size = self.image_size
        # 如果image的高和宽不等于final_size，则进行resize
        if image.size()[0] != final_size or image.size()[1] != final_size:
            Transform.append(T.Resize((final_size, final_size)))
            Transform_GT.append(T.Resize((final_size, final_size)))
            Transform_contour.append(T.Resize((final_size, final_size)))
            Transform_dist.append(T.Resize((final_size, final_size)))
            if self.load_preseg:
                Transform.append(T.Resize((final_size, final_size)))


        # Transform.append(T.Resize((final_size, final_size), interpolation=Image.BICUBIC))
        # Transform_GT.append(T.Resize((final_size, final_size), interpolation=Image.NEAREST))
        # Transform_contour.append(T.Resize((final_size, final_size), interpolation=Image.NEAREST))
        # Transform_dist.append(T.Resize((final_size, final_size), interpolation=Image.NEAREST))

        # Transform.append(T.ToTensor())
        # Transform_GT.append(T.ToTensor())
        # Transform_contour.append(T.ToTensor())
        # Transform_dist.append(T.ToTensor())

        if image.mode == 'RGB':
            Transform.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        # print('tensor has be normalized')

        Transform = T.Compose(Transform)
        Transform_GT = T.Compose(Transform_GT)
        Transform_contour = T.Compose(Transform_contour)
        Transform_dist = T.Compose(Transform_dist)

        image = Transform(image)
        GT = Transform_GT(GT)
        contour = Transform_contour(contour)
        dist = Transform_dist(dist)
        if self.load_preseg:
            seg = Transform_GT(seg)
        # print(GT.shape)
        # 如果是rgb,则需要
        # Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # image = Norm_(image)
        if self.load_preseg:
            # print('load seg')
            return image_path, image, GT, contour, dist, seg
        else:
            return image_path, image, GT, contour, dist, class_item

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


class ImageFolder_new_difficult(data.Dataset):
    def __init__(self, seg_list, GT_list, class_list, contour_list, dist_list, image_list, load_preseg=False, image_size=512, mode='train',
                 augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        # self.root = root
        # GT : Ground Truth
        # self.GT_paths = os.path.join(root, 'p_mask')
        self.GT_paths = GT_list
        self.image_paths = image_list
        self.contour_paths = contour_list
        self.dist_paths = dist_list
        self.class_list = class_list

        self.load_preseg = load_preseg
        if self.load_preseg:
            self.seg_paths = seg_list

        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270, 45, 135, 215, 305]
        # self.RotationDegree = [0,90,180,270,45]
        self.augmentation_prob = augmentation_prob
        self.resize_range = [520, 560]
        self.CropRange = [400, 519]  # 注意,上界貌似不能大于resize的下界,待验证

    # print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        contour_path = self.contour_paths[index]
        dist_path = self.dist_paths[index]
        filename = image_path.split('/')[-1]
        GT_path = self.GT_paths[index]
        class_item = self.class_list[index]
        # 将class_item转化为tensor
        class_item = torch.tensor(int(class_item))


        image_o = image = Image.open(image_path)
        contour_o = contour = Image.open(contour_path)
        dist_o = dist = Image.open(dist_path)
        GT_o = GT = Image.open(GT_path)

        if self.load_preseg:
            seg_path = self.seg_paths[index]
            seg_o = seg = Image.open(seg_path)

        aspect_ratio = image.size[1] / image.size[0]

        Transform = []
        Transform_GT = []  # 注意,GT的插值需要最近邻nearest,但是采取非线性插值可能有奇效
        Transform_contour = []
        Transform_dist = []


        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:

            # ----------------------------------- my TTA ------------------------------------

            if self.load_preseg:
                [image, GT, contour, dist, seg] = img_mask_aug.data_aug(image, GT, contour, dist, seg)
            else:
                [image, GT, contour, dist] = img_mask_aug.data_aug(image, GT, contour, dist)

            image = Image.fromarray(image)
            GT = Image.fromarray(GT)
            contour = Image.fromarray(contour)
            dist = Image.fromarray(dist)
            if self.load_preseg:
                seg = Image.fromarray(seg)

        final_size = self.image_size
        Transform.append(T.Resize((final_size, final_size), interpolation=Image.BICUBIC))
        Transform_GT.append(T.Resize((final_size, final_size), interpolation=Image.NEAREST))
        Transform_contour.append(T.Resize((final_size, final_size), interpolation=Image.NEAREST))
        Transform_dist.append(T.Resize((final_size, final_size), interpolation=Image.NEAREST))

        Transform.append(T.ToTensor())
        Transform_GT.append(T.ToTensor())
        Transform_contour.append(T.ToTensor())
        Transform_dist.append(T.ToTensor())

        Transform = T.Compose(Transform)
        Transform_GT = T.Compose(Transform_GT)
        Transform_contour = T.Compose(Transform_contour)
        Transform_dist = T.Compose(Transform_dist)


        image = Transform(image)
        GT = Transform_GT(GT)
        contour = Transform_contour(contour)
        dist = Transform_dist(dist)
        if self.load_preseg:
            seg = Transform_GT(seg)
        # print(GT.shape)
        # 如果是rgb,则需要
        # Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # image = Norm_(image)

        if self.load_preseg:
            # print('load seg')
            return image_path, image, GT, contour, dist, class_item, seg
        else:
            return image_path, image, GT, contour, dist, class_item

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def worker_init_fn(worker_id):
    torch.cuda.empty_cache()  # 在每个子进程中调用torch.cuda.empty_cache()命令

def get_loader(seg_list,
               GT_list,
               class_list,
               image_list,
               contour_list,
               dist_list,
               image_size,
               batch_size,
               load_preseg=False,
               num_workers=2,
               mode='train',
               augmentation_prob=0.4,
               device=None):
    """Builds and returns Dataloader."""

    dataset = ImageFolder_new(load_preseg=load_preseg,
                              seg_list=seg_list,
                              GT_list=GT_list,
                              class_list=class_list,
                              contour_list=contour_list,
                              dist_list=dist_list,
                              image_list=image_list,
                              image_size=image_size,
                              mode=mode,
                              augmentation_prob=augmentation_prob,
                              device=device)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn,
                                  # pin_memory=True
                                  )
    return data_loader


def get_loader_difficult(seg_list,
                         GT_list,
                         class_list,
                         image_list,
                         contour_list,
                         dist_list,
                         image_size,
                         batch_size,
                         load_preseg=False,
                         num_workers=2,
                         mode='train',
                         augmentation_prob=0.4):
    """Builds and returns Dataloader."""
    dataset = ImageFolder_new_difficult(load_preseg=load_preseg,
                                        seg_list=seg_list,
                                        GT_list=GT_list,
                                        class_list=class_list,
                                        contour_list=contour_list,
                                        dist_list=dist_list,
                                        image_list=image_list,
                                        image_size=image_size,
                                        mode=mode,
                                        augmentation_prob=augmentation_prob)
    # print(char_color('@,,@   doing with difficult augmentation'))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  pin_memory=True
                                  )
    return data_loader


def DrawSavePic(img_file_name, inputs, targets1, targets2, targets3, train_pic_list):
    # 检测文件夹是否存在
    if not os.path.exists(train_pic_list):
        os.makedirs(train_pic_list)

    # 确定批次大小和图像数量
    batch_size = inputs.size(0)
    num_images = batch_size

    # 确定子图布局
    fig, axes = plt.subplots(1, 2, figsize=(8, 8), constrained_layout=True)

    # 遍历每个样本
    for i in range(num_images):
        # 获取当前样本的输入和目标
        input_img = inputs[i].squeeze().cpu().numpy()
        target1_img = targets1[i].squeeze().cpu().numpy()

        # 绘制当前样本的输入和目标
        axes[0].imshow(input_img, cmap='gray')
        axes[1].imshow(target1_img, cmap='gray')

        # 获取当前图像的文件名，'../class_out/stage1/p_image\\497.png'
        img_file_names = img_file_name[i]
        # 获取497.png
        img_file_names = img_file_names.split(sep)[-1]
        # 获取数字497
        NUM = img_file_names.split(sep)[0]

        # 设置当前子图的标题
        axes[0].set_title(f'img {NUM}')
        axes[1].set_title(f'GT {NUM}')
        # axes[1, 0].set_title(f'contour {i+1}')
        # axes[1, 1].set_title(f'dist {i+1}')

        # 重新创建个文件名，'train_pic_497.png'
        img_file_names = 'train_pic_' + img_file_names

        # 保存图片
        plt.savefig(train_pic_list + img_file_names)
        plt.close()
        print('save pic:', img_file_names)


if __name__ == '__main__':
    import class_divide
    import img_mask_aug

    csv_file = '../class_out/train.csv'
    fold_K = 5
    fold_idx = 1
    image_size = 224
    batch_size = 30
    testbs = 1
    num_workers = 0
    validate_flag = True
    augmentation_prob = 0.8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sep = os.sep  # os.sep根据你所处的平台，自动采用相应的分隔符号
    if validate_flag:
        train, valid, test = class_divide.get_fold_filelist(csv_file, K=fold_K, fold=fold_idx, validation=True)
    else:
        train, test = class_divide.get_fold_filelist(csv_file, K=fold_K, fold=fold_idx)
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

    filepath_img = '../class_out/stage1/p_image'
    filepath_mask = '../class_out/stage1/p_mask'
    filepath_contour = '../class_out/stage1/p_contour'
    filepath_dist = '../class_out/stage1/p_distance_D1'

    train_list = [filepath_img + sep + i[0] for i in train]
    train_list_GT = [filepath_mask + sep + i[0] for i in train]
    trian_list_contour = [filepath_contour + sep + i[0] for i in train]
    train_list_dist = [filepath_dist + sep + i[0] for i in train]
    train_class_list_GT = [i[1] for i in train]

    valid_list = [filepath_img + sep + i[0] for i in valid]
    valid_list_GT = [filepath_mask + sep + i[0] for i in valid]
    valid_list_contour = [filepath_contour + sep + i[0] for i in valid]
    valid_list_dist = [filepath_dist + sep + i[0] for i in valid]
    valid_class_list_GT = [i[1] for i in valid]

    test_list = [filepath_img + sep + i[0] for i in test]
    test_list_GT = [filepath_mask + sep + i[0] for i in test]
    test_list_contour = [filepath_contour + sep + i[0] for i in test]
    test_list_dist = [filepath_dist + sep + i[0] for i in test]
    test_class_list_GT = [i[1] for i in test]

    print("images count in train:{}".format(len(train_list)))
    print("images count in valid:{}".format(len(valid_list)))
    print("images count in test :{}".format(len(test_list)))

    # 将train_list, valid_list, test_list保存到一个txt文件中
    train_list_txt = '../foldinfo/train_list_' + str(fold_idx) + '.txt'
    valid_list_txt = '../foldinfo/valid_list_' + str(fold_idx) + '.txt'
    test_list_txt = '../foldinfo/test_list_' + str(fold_idx) + '.txt'
    utils.WriteIntoTxt(train_list, train_list_txt)
    utils.WriteIntoTxt(valid_list, valid_list_txt)
    utils.WriteIntoTxt(test_list, test_list_txt)
    train_loader = get_loader(seg_list=None,
                              GT_list=train_list_GT,
                              class_list=train_class_list_GT,
                              image_list=train_list,
                              contour_list=trian_list_contour,
                              dist_list=train_list_dist,
                              image_size=image_size,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              mode='train',
                              augmentation_prob=augmentation_prob,
                              device=device)

    valid_loader = get_loader(seg_list=None,
                              GT_list=valid_list_GT,
                              class_list=valid_class_list_GT,
                              image_list=valid_list,
                              contour_list=valid_list_contour,
                              dist_list=valid_list_dist,
                              image_size=image_size,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              mode='val',
                              augmentation_prob=0., )

    test_loader = get_loader(seg_list=None,
                             GT_list=test_list_GT,
                             class_list=test_class_list_GT,
                             image_list=test_list,
                             contour_list=test_list_contour,
                             dist_list=test_list_dist,
                             image_size=image_size,
                             batch_size=testbs,
                             num_workers=num_workers,
                             mode='test',
                             augmentation_prob=0., )

    # 接下来模拟训练过程
    # 1. 定义一个简单的模型
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 2)
    model = model.to(device)

    # 2. 定义一个简单的损失函数
    criterion = nn.CrossEntropyLoss()
    # 3. 定义一个简单的优化器
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_pic_list = '../foldinfo/train_list_/'
    # 开始训练
    datas = train_loader
    for epoch in range(1, 10):
        for i, data in enumerate(datas, 0):
            (img_file_name, inputs, targets1, targets2, targets3, targets4) = data
            if epoch == 1:
                # 保存图片
                DrawSavePic(img_file_name, inputs, targets1, targets2, targets3, train_pic_list)

            # 睡眠0.1秒，模拟训练过程
            time.sleep(0.1)
            # print 一些随机浮点数，模拟训练过程
            print(epoch, i, random.random())
            # 将inputs, targets1, targets2, targets3利用matplotlib画出来

            # # 在每次迭代结束时调用torch.cuda.empty_cache()命令
            # torch.cuda.empty_cache()








