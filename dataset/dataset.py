import glob

import torch
import numpy as np
import cv2
from PIL import Image, ImageFile
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io


class DatasetImageMaskContourDist(Dataset):

    # dataset_type(cup,disc,polyp),
    # distance_type(dist_mask,dist_contour,dist_signed)

    def __init__(self, file_names, distance_type, normal_flag):

        self.file_names = file_names    # ./train_path/fold/fold1/train/images/*.png
        self.distance_type = distance_type
        self.normal_flag = normal_flag

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        img_basename = os.path.basename(img_file_name)
        mask_file_path = img_file_name.replace("images", "mask")
        contour_file_path = img_file_name.replace("images", "contour")
        dist_file_path = img_file_name.replace("images", self.distance_type)
        cls_path = r'../train_path/train.csv'

        image = load_image(img_file_name, normal_flag=self.normal_flag)
        mask = load_mask(mask_file_path)
        # todo:还有图像分类的任务，新添加一个标签叫cls,cls有三种值，0，1，2，分别代表良性，恶性，正常
        cls = load_cls(cls_path, img_basename)
        contour = load_contour(contour_file_path)
        dist = load_distance(dist_file_path, self.distance_type)

        return img_file_name, image, mask, contour, dist, cls


def load_cls(path, name):
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if name in line:
                cls = line.split(',')[1]
                # 将cls转换为tensor
                cls = torch.tensor(int(cls))
                return cls

def load_image(path, normal_flag):
    img = Image.open(path)
    img_path = path.split('\\')[0]

    if normal_flag:
        mean, std = calculate_mean_std(img_path)
        data_transforms = transforms.Compose(   # Compose 将多个变换组合在一起，包括transforms，ToTensor，Normalize等。
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                lambda x: torch.stack([x[0], x[0], x[0]], dim=0),
            ]
        )
    else:
        data_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                lambda x: torch.stack([x[0], x[0], x[0]], dim=0),
            ]
        )
    # 相当于先resize，再转换为tensor，最后归一化，归一化的参数是ImageNet的均值和标准差
    img = data_transforms(img)

    return img

def calculate_mean_std(dataset_path):
    file_names = glob.glob(os.path.join(dataset_path, '*.png'))
    mean_list = []
    std_list = []

    for file_name in file_names:
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        # 检测是否存在分母为0的情况
        if np.std(img) == 0:
            continue
        mean_list.append(np.mean(img))
        std_list.append(np.std(img))

    mean = np.mean(mean_list)
    std = np.mean(std_list)

    return mean, std



def load_mask(path):
    # 读取mask，mask是灰度图，像素点值为0或255
    mask = cv2.imread(path, 0)  # 0表示灰度图
    mask[mask == 255] = 1   # 二值化，将255的像素点变为1，其余的本身就是0，则不变

    return torch.from_numpy(np.expand_dims(mask, 0)).long()


# contour是mask的边缘，所以contour的像素点值只有0和1
def load_contour(path):

    contour = cv2.imread(path, 0)
    contour[contour == 255] = 1

    return torch.from_numpy(np.expand_dims(contour, 0)).long()

# distance是
def load_distance(path, distance_type):
    # 读取PNG图像文件
    img = Image.open(path)
    # 转换为NumPy数组
    distance_array = np.array(img)

    # if distance_type == "dist_mask":
    #     # path = path.replace("image", "dist_mask").replace("jpg", "mat")
    #     dist = io.loadmat(path)["mask_dist"]
    #
    # if distance_type == "dist_contour":
    #     # path = path.replace("image", "dist_contour").replace("jpg", "mat")
    #     dist = io.loadmat(path)["contour_dist"]
    #
    # if distance_type == "dist_signed":
    #     # path = path.replace("image", "dist_signed").replace("jpg", "mat")
    #     dist = io.loadmat(path)["dist_norm"]

    return torch.from_numpy(np.expand_dims(distance_array, 0)).float()


