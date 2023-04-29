import random

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
import warnings
warnings.filterwarnings('ignore', message='Argument \'interpolation\' of type int is deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.')
# from torchvision.transforms import InterpolationMode


from dataset.img_mask_aug import data_aug


class ImageFolder_new(data.Dataset):
    def __init__(self, seg_list, GT_list, class_list, image_list, image_size=512, mode='train', augmentation_prob=0.4,
                 load_preseg=False):
        """Initializes image paths and preprocessing module."""
        # self.root = root

        # GT : Ground Truth
        # self.GT_paths = os.path.join(root, 'p_mask')
        self.GT_paths = GT_list
        self.image_paths = image_list
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
        GT_path = self.GT_paths[index]
        class_item = self.class_list[index]

        filename = image_path.split('/')[-1]
        # GT_path = os.path.join(self.GT_paths, filename)

        image_o = image = Image.open(image_path)
        GT_o = GT = Image.open(GT_path)

        if self.load_preseg:
            seg_path = self.seg_paths[index]
            seg_o = seg = Image.open(seg_path)

        aspect_ratio = image.size[1] / image.size[0]

        Transform = []
        Transform_GT = []  # 注意,GT的插值需要最近邻nearest,但是采取非线性插值可能有奇效

        ResizeRange = random.randint(self.resize_range[0], self.resize_range[1])

        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            Transform.append(
                T.Resize((int(ResizeRange * aspect_ratio), ResizeRange), interpolation=Image.BICUBIC))  # 双三次
            Transform_GT.append(
                T.Resize((int(ResizeRange * aspect_ratio), ResizeRange), interpolation=Image.NEAREST))  # 最近邻

            RotationDegree = random.randint(0, 3)
            RotationDegree = self.RotationDegree[RotationDegree]
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1 / aspect_ratio

            Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))
            Transform_GT.append(T.RandomRotation((RotationDegree, RotationDegree)))

            # 在大旋转间隔基础上,微小调整旋转角度
            RotationRange = random.randint(-10, 10)
            Transform.append(T.RandomRotation((RotationRange, RotationRange)))
            Transform_GT.append(T.RandomRotation((RotationRange, RotationRange)))

            CropRange = random.randint(self.CropRange[0], self.CropRange[1])
            Transform.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
            Transform_GT.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))

            Transform = T.Compose(Transform)
            Transform_GT = T.Compose(Transform_GT)

            image = Transform(image)
            GT = Transform_GT(GT)
            if self.load_preseg:
                seg = Transform_GT(seg)

            # crop
            ShiftRange_left = random.randint(0, 20)
            ShiftRange_upper = random.randint(0, 20)
            ShiftRange_right = image.size[0] - random.randint(0, 20)
            ShiftRange_lower = image.size[1] - random.randint(0, 20)
            image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
            GT = GT.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
            if self.load_preseg:
                seg = seg.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))

            # flip
            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)
                if self.load_preseg:
                    seg = F.hflip(seg)
            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)
                if self.load_preseg:
                    seg = F.vflip(seg)

            # Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)
            #
            # image = Transform(image)

            Transform = []
            Transform_GT = []

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
        Transform.append(T.Resize((final_size, final_size), interpolation=Image.BICUBIC))
        Transform_GT.append(T.Resize((final_size, final_size), interpolation=Image.NEAREST))

        Transform.append(T.ToTensor())
        Transform_GT.append(T.ToTensor())

        if image.mode == 'RGB':
            Transform.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        # print('tensor has be normalized')

        Transform = T.Compose(Transform)
        Transform_GT = T.Compose(Transform_GT)

        image = Transform(image)
        GT = Transform_GT(GT)
        if self.load_preseg:
            seg = Transform_GT(seg)
        # print(GT.shape)
        # 如果是rgb,则需要
        # Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # image = Norm_(image)
        if self.load_preseg:
            # print('load seg')
            return image_path, image, GT, seg
        else:
            return image_path, image, GT, class_item

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
                [image, GT, contour, dist, seg] = data_aug(image, GT, contour, dist, seg)
            else:
                [image, GT, contour, dist] = data_aug(image, GT, contour, dist)

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


def get_loader(seg_list, GT_list, class_list, image_list,
               image_size, batch_size,
               load_preseg=False, num_workers=2, mode='train', augmentation_prob=0.4):
    """Builds and returns Dataloader."""

    dataset = ImageFolder_new(load_preseg=load_preseg,
                              seg_list=seg_list,
                              GT_list=GT_list,
                              class_list=class_list,
                              image_list=image_list,
                              image_size=image_size,
                              mode=mode,
                              augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True)
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
                                  drop_last=False)
    return data_loader
