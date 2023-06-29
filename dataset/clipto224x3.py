# 这个主要是用来对图片进行裁切的，裁切成224x224的大小，然后保存到指定的文件夹中
# 图片包括原图和mask图
# 裁切分为两部分，一个是裁切有结节的图片，一个是裁切没有结节的图片
# 对于裁切有结节的图片，裁切的时候，会将结节的mask也一起裁切下来，如果结节小，则先确定结节中心点，然后以结节中心点为中心，裁切224x224的图片
# 如果结节大，则先将结节mask进行腐蚀，然后再确定结节中心点，然后以结节中心点为中心，裁切224x224的图片

import math
import os

import cv2
import numpy as np
import csv


# 定义一个函数，实现计算某个水平方向上能放下的裁切框的个数
def get_num(img_size, clip_size):
    if img_size < clip_size:
        raise Exception("图片的尺寸小于裁切的尺寸")
    # 计算方法是 math.ceil(img_size / clip_size)
    num = math.ceil(img_size / clip_size)

    return num


# 定义一个函数，计算某个方向上的裁切后每个sub_img之间的交叉区域的宽度
def get_cross(img_size, clip_size, num):
    if img_size < clip_size:
        raise Exception("图片的尺寸小于裁切的尺寸")
    # 计算方法是 (num * clip_size - img_size) / (num - 1)，并向下取整
    if num == 1:
        cross = 0
    else:
        cross = math.floor((num * clip_size - img_size) / (num - 1))

    return cross


# 定义一个函数，用于计算clip的窗口沿着某个方向每次移动的距离move
def get_move(clip_size, cross):
    # 计算方法是 clip_size - cross
    move = clip_size - cross

    return move


# 定义一个函数，用于获取num_x, num_y, cross_x, cross_y
def get_num_cross(img_size, clip_size):
    num_x = get_num(img_size[0], clip_size[0])
    num_y = get_num(img_size[1], clip_size[1])
    cross_x = get_cross(img_size[0], clip_size[0], num_x)
    cross_y = get_cross(img_size[1], clip_size[1], num_y)
    # 需要注意的是，cross_x和cross_y是有范围的，他必须大于20，不能交叉太少，因为交叉太少，那么sub_img之间的交叉区域就太小了，
    # 这样就会导致sub_img之间的交叉区域的特征不明显
    if cross_x < 20:
        # 如果cross_x小于20，则将num_x加1，这样sub_img数量就会增大，从而cross_x也会增大
        num_x += 1
        # 然后重新计算cross_x
        cross_x = get_cross(img_size[0], clip_size[0], num_x)

    if cross_y < 20:
        # 如果cross_y小于20，则将num_y加1，这样sub_img数量就会增大，从而cross_y也会增大
        num_y += 1
        # 然后重新计算cross_y
        cross_y = get_cross(img_size[1], clip_size[1], num_y)

    return num_x, num_y, cross_x, cross_y


# 定义一个函数，用于获取x，y方向上的移动距离
def get_move_xy(clip_size, cross_x, cross_y):
    move_x = get_move(clip_size[0], cross_x)
    move_y = get_move(clip_size[1], cross_y)

    return move_x, move_y


# 获取总共的sub_img的数量
def get_total_num(num_x, num_y):
    total_num = num_x * num_y

    return total_num


# 定义一个函数，用于针对某种类型的图片，裁切出sub_img的算法
def clip_img(temp_img, clip_size):
    # 初始化起始的x坐标和y坐标
    cox = 0
    coy = 0
    # 获取图片的尺寸
    img_size = temp_img.shape
    # 如果图片的尺寸小于裁切的尺寸，则将其放大到裁切的尺寸
    if img_size[0] < clip_size[0]:
        if img_size[1] < clip_size[1]:
            # 宽和高都小于裁切的尺寸，则将图片放大到裁切的尺寸
            temp_img = cv2.resize(temp_img, clip_size, interpolation=cv2.INTER_CUBIC)
        else:
            # 宽小于裁切的尺寸，高大于裁切的尺寸，则将图片的宽放大到裁切的尺寸
            temp_img = cv2.resize(temp_img, (clip_size[0], img_size[1]), interpolation=cv2.INTER_CUBIC)
    else:
        if img_size[1] < clip_size[1]:
            # 宽大于裁切的尺寸，高小于裁切的尺寸，则将图片的高放大到裁切的尺寸
            temp_img = cv2.resize(temp_img, (img_size[0], clip_size[1]), interpolation=cv2.INTER_CUBIC)
    # 重新获取图片的尺寸
    img_size = temp_img.shape
    # 获取num_x, num_y, cross_x, cross_y
    num_x, num_y, cross_x, cross_y = get_num_cross(img_size, clip_size)
    # 获取总共的sub_img的数量
    total_num = get_total_num(num_x, num_y)
    # 获取x，y方向上的移动距离
    move_x, move_y = get_move_xy(clip_size, cross_x, cross_y)
    # 初始化一个列表，用于存放sub_img
    sub_img_list = []

    # 开始裁切图片
    for i in range(num_x):  # i = 0, 1, 2, 3, 4, 5, 6, 7, 8, ... num_x - 1
        # 表示第i次裁切的起始x坐标，当i == 0 时，cox = 0，当i == 1时，cox = move_x，当i == 2时，cox = 2 * move_x，以此类推
        cox = i * move_x
        for j in range(num_y):  # j = 0, 1, 2, 3, 4, 5, 6, 7, 8, ... num_y - 1
            # 表示第j次裁切的起始y坐标，当j == 0 时，coy = 0，当j == 1时，coy = move_y，当j == 2时，coy = 2 * move_y，以此类推
            coy = j * move_y
            # 裁切图片
            sub_img = temp_img[cox: cox + clip_size[1], coy: coy + clip_size[0]]
            # 将裁切出来的sub_img添加到列表中
            sub_img_list.append(sub_img)

    return sub_img_list


# 定义一个函数，用于利用clip_img对img，mask，contour, distance进行裁切
def clip_img_mask_contour_distance(img, mask, contour, distance, clip_size):
    # 以下这些列表中的元素都是一一对应的
    # 裁切img
    img_list = clip_img(img, clip_size)
    # 裁切mask
    mask_list = clip_img(mask, clip_size)
    # 裁切contour
    contour_list = clip_img(contour, clip_size)
    # 裁切distance
    distance_list = clip_img(distance, clip_size)

    return img_list, mask_list, contour_list, distance_list


def readCsv(csvfname):
    # read csv to list of lists
    print(csvfname)
    with open(csvfname, 'r', ) as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines


# 针对mask_list中的每个sub_mask，查看其中是否含有白色的像素点，如果有，则说明是包含结节的sub_mask，则label不变，如果没有，则将其label置为2.
def get_label(mask_list, old_label):
    # 初始化一个列表，用于存放label
    label_list = []
    # 开始遍历mask_list
    for sub_mask in mask_list:
        # 如果sub_mask中含有白色的像素点，则说明是包含结节的sub_mask，则label不变
        if np.max(sub_mask) == 255:
            label_list.append(old_label)
        # 如果sub_mask中不含有白色的像素点，则说明不是包含结节的sub_mask，则label置为2
        else:
            label_list.append(2)

    return label_list


# 定义一个函数， 用于计算nodule_num 的像素数目，
def get_nodule_num(mask_list):
    # 初始化一个列表，用于存放nodule_num
    nodule_num_list = []
    # 开始遍历mask_list
    for sub_mask in mask_list:
        # 获取sub_mask中白色像素点的个数
        nodule_num = np.sum(sub_mask == 255)
        # 将nodule_num添加到列表中
        nodule_num_list.append(nodule_num)

    return nodule_num_list


# 定义一个函数，裁切整个filepath_img，filepath_mask，filepath_contour，filepath_distance，
# 然后将裁切出来的sub_img，sub_mask，sub_contour，sub_distance保存到指定的文件夹中
def clip_img_mask_contour_distance_save(new_img_name, filepath_img, filepath_mask, filepath_contour, filepath_distance,
                                        savepath_img, old_label, savepath_mask, savepath_contour, savepath_distance,
                                        clip_size, savepath_csv, nodule_num):
    '''
    :param new_img_name: 用于记录编号
    :param filepath_img: 原始图片的路径, 单张图片
    :param filepath_mask: 原始mask的路径, 单张图片
    :param filepath_contour: 原始contour的路径, 单张图片
    :param filepath_distance: 原始distance的路径, 单张图片
    :param savepath_img: 裁切后的图片的保存路径
    :param old_label: 原始图片的label
    :param savepath_mask: 裁切后的mask的保存路径
    :param savepath_contour: 裁切后的contour的保存路径
    :param savepath_distance: 裁切后的distance的保存路径
    :param clip_size: 裁切的尺寸
    :param savepath_csv: 保存csv文件的路径
    :param nodule_num: 保存csv文件的编号
    :return: count: 用于记录裁切的图片的数量,并作为最后裁切出来的图片的编号
    '''
    # 读取img,是RGB的
    img = cv2.imread(filepath_img, cv2.IMREAD_COLOR)  # IMREAD_COLOR是读取彩色图像，该图像为三通道的彩色图像，忽略alpha通道
    # 读取mask
    mask = cv2.imread(filepath_mask, cv2.IMREAD_GRAYSCALE)
    # 读取contour
    contour = cv2.imread(filepath_contour, cv2.IMREAD_GRAYSCALE)
    # 读取distance
    distance = cv2.imread(filepath_distance, cv2.IMREAD_GRAYSCALE)

    # 裁切img，mask，contour，distance
    img_list, mask_list, contour_list, distance_list = clip_img_mask_contour_distance(img, mask, contour, distance,
                                                                                      clip_size)

    # 获取mask_list中每个sub_mask的label
    label_list = get_label(mask_list, old_label)

    # 获取mask_list中每个sub_mask的nodule_num
    nodule_num_list = get_nodule_num(mask_list)

    # 开始遍历img_list，mask_list，contour_list，distance_list，label_list
    # 保存，但是原始图片有799张，都是排好序的，那每一张原始图片裁切出来的子图片也要按照顺序保存，所以要在这里加一个计数器
    for sub_img, sub_mask, sub_contour, sub_distance, label, nodule_num in zip(img_list, mask_list, contour_list,
                                                                               distance_list,
                                                                               label_list,
                                                                               nodule_num_list):
        # 保存裁切出来的sub_img
        cv2.imwrite(savepath_img + '/' + str(new_img_name) + '.png', sub_img)
        # 保存裁切出来的sub_mask
        cv2.imwrite(savepath_mask + '/' + str(new_img_name) + '.png', sub_mask)
        # 保存裁切出来的sub_contour
        cv2.imwrite(savepath_contour + '/' + str(new_img_name) + '.png', sub_contour)
        # 保存裁切出来的sub_distance
        cv2.imwrite(savepath_distance + '/' + str(new_img_name) + '.png', sub_distance)

        # 若label为0或者1，则说明含有结节，那么需要重新寄宿包含结节的图像中的nodule_num，因为这个nodule_num 代表的是
        # 图像中为结节的像素数目。所以需要重新计算

        # ID 使用ID.png
        # CATE 使用label
        # size 使用nodulenum
        with open(savepath_csv, 'a') as f:
            f.write(str(new_img_name) + '.png' + ',' + str(label) + ',' + str(nodule_num) + '\n')
        # 计数器加1
        new_img_name += 1

    return new_img_name


# 创建一个函数，实现对目标文件夹列表中的文件夹的检测，如果不存在就创建
def create_dir(dir_list):
    """
    :param dir_list: 目标文件夹列表
    :return: None
    """
    for dir in dir_list:
        if not os.path.exists(dir):
            os.mkdir(dir)


def min_maxsize(img_path):
    """
    :param img_path: 199张原始图片的路径
    :return: min_size, max_size
    """  # 创建一个列表，用于存放所有图片的尺寸
    size_list_h = []
    size_list_w = []
    # 读取img_path中的所有图片
    img_list = os.listdir(img_path)
    # 开始遍历img_list
    for img in img_list:
        # 读取图片,图片是三通道的，只需要读取一个通道就可以了
        Img = cv2.imread(img_path + '/' + img, cv2.IMREAD_COLOR)
        # 获取第一个通道的的高和宽
        h, w = Img[:, :, 0].shape
        # 将h和w添加到size_list_h和size_list_w中
        size_list_h.append(h)
        size_list_w.append(w)
    # 获取size_list_h和size_list_w中的最大值和最小值
    min_size_h = min(size_list_h)
    min_size_w = min(size_list_w)  # 2103495
    # 获取最小值的索引
    min_index_h = size_list_h.index(min_size_h)
    min_index_w = size_list_w.index(min_size_w)
    max_size_h = max(size_list_h)
    max_size_w = max(size_list_w)
    # 对size_list_h中的值进行排序
    size_list_h.sort()
    size_list_w.sort()
    # 将size_list_h和size_list_w中小于224的统计个数
    count_h = 0
    count_w = 0
    for i in size_list_h:
        if i < 224:
            count_h += 1
    for i in size_list_w:
        if i < 224:
            count_w += 1
    print('size_list_h中小于224的个数为：', count_h)
    print('size_list_w中小于224的个数为：', count_w)


def main():
    filepath_img = '../class_out/clip_dataset/p_image'
    filepath_mask = '../class_out/clip_dataset/p_mask'
    filepath_contour = '../class_out/clip_dataset/p_contour'
    filepath_dist = '../class_out/clip_dataset/p_distance_D1'
    filepath_csv = '../class_out/train.csv'

    # 用一个函数min_maxsize函数读取filepath_img中的所有图片，输出宽和高的最大值和最小值，并统计小于224的个数，结果只有一个
    min_maxsize(filepath_img)

    savepath_img = '../class_out/clip_dataset/clip_image'
    savepath_mask = '../class_out/clip_dataset/clip_mask'
    savepath_contour = '../class_out/clip_dataset/clip_contour'
    savepath_dist = '../class_out/clip_dataset/clip_distance_D1'
    savepath_csv = '../class_out/clip_dataset/clip_train.csv'
    pathlist = [savepath_img, savepath_mask, savepath_contour, savepath_dist]
    create_dir(pathlist)

    clip_size = [224, 224]
    # 读取csv文件
    CTcsvlines = readCsv(filepath_csv)

    if os.path.exists(savepath_csv):
        # 如果文件存在，就删除再创建
        os.remove(savepath_csv)
        with open(savepath_csv, 'w') as f:
            f.write('ID' + ',' + 'CATE' + ',' + 'size' + '\n')
    else:
        with open(savepath_csv, 'w') as f:
            f.write('ID' + ',' + 'CATE' + ',' + 'size' + '\n')

    # 设置起始的图片名字的计数器为1
    count = 1
    # 开始遍历csv文件中的每一行,csv表头是['img_name', 'label','nodule_num']
    # 开始之前去除第一行的表头
    CTcsvlines.pop(0)
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
        # 裁切img，mask，contour，distance
        count = clip_img_mask_contour_distance_save(count, filepath_img_l, filepath_mask_l, filepath_contour_l,
                                                    filepath_dist_l,
                                                    savepath_img, label, savepath_mask, savepath_contour, savepath_dist,
                                                    clip_size, savepath_csv, nodule_num)
        print("已完成第{}张图片的裁切".format(count))

    print('裁切完成！')


if __name__ == '__main__':
    main()
