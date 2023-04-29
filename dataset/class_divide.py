from sklearn.model_selection import StratifiedKFold
import csv

def readCsv(csvfname):
    # read csv to list of lists
    print(csvfname)
    with open(csvfname, 'r',) as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines

def get_fold_filelist(csv_file, K=3, fold=1, random_state=2020, validation=False, validation_r = 0.2):
    """
    获取分折结果的API（基于size分3层的类别平衡分折）
    :param csv_file: 带有ID、CATE、size的文件
    :param K: 分折折数
    :param fold: 返回第几折,从1开始
    :param random_state: 随机数种子, 固定后每次实验分折相同(注意,sklearn切换版本可能会导致相同随机数种子产生不同分折结果)
    :param validation: 是否需要验证集（从训练集随机抽取部分数据当作验证集）
    :param validation_r: 抽取出验证集占训练集的比例
    :return: train和test的list，带有label和size
    """

    CTcsvlines = readCsv(csv_file)
    header = CTcsvlines[0]
    print('header', header)
    nodules = CTcsvlines[1:]

    # 提取size的三分点
    sizeall = [int(i[2]) for i in nodules]
    sizeall.sort()  # 按升序排列
    low_mid_thre = sizeall[int(len(sizeall)*1/3)]
    mid_high_thre = sizeall[int(len(sizeall)*2/3)]

    # 根据size三分位数分为low，mid，high三组

    low_size_list = [i for i in nodules if int(i[2]) < low_mid_thre]
    mid_size_list = [i for i in nodules if int(i[2]) < mid_high_thre and int(i[2]) >= low_mid_thre]
    high_size_list = [i for i in nodules if int(i[2]) >= mid_high_thre]

    # 将lable划分为三组
    low_label = [int(i[1]) for i in low_size_list]
    mid_label = [int(i[1]) for i in mid_size_list]
    high_label = [int(i[1]) for i in high_size_list]


    low_fold_train = []
    low_fold_test = []

    mid_fold_train = []
    mid_fold_test = []

    high_fold_train = []
    high_fold_test = []

    sfolder = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=True)
    for train, test in sfolder.split(low_label, low_label):
        low_fold_train.append([low_size_list[i] for i in train])
        low_fold_test.append([low_size_list[i] for i in test])

    sfolder = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=True)
    for train, test in sfolder.split(mid_label, mid_label):
        mid_fold_train.append([mid_size_list[i] for i in train])
        mid_fold_test.append([mid_size_list[i] for i in test])

    sfolder = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=True)
    for train, test in sfolder.split(high_label, high_label):
        high_fold_train.append([high_size_list[i] for i in train])
        high_fold_test.append([high_size_list[i] for i in test])

    if validation is False: # 不设置验证集，则直接返回
        train_set = low_fold_train[fold-1]+mid_fold_train[fold-1]+high_fold_train[fold-1]
        test_set = low_fold_test[fold-1]+mid_fold_test[fold-1]+high_fold_test[fold-1]
        return [train_set, test_set]
    else:  # 设置验证集合，则从训练集“类别 且 size平衡地”抽取一定数量样本做验证集
        # 分离第fold折各size分层的正负样本
        low_fold_train_p = [i for i in low_fold_train[fold-1] if int(i[1]) == 1]
        low_fold_train_n = [i for i in low_fold_train[fold-1] if int(i[1]) == 0]

        mid_fold_train_p = [i for i in mid_fold_train[fold-1] if int(i[1]) == 1]
        mid_fold_train_n = [i for i in mid_fold_train[fold-1] if int(i[1]) == 0]

        high_fold_train_p = [i for i in high_fold_train[fold-1] if int(i[1]) == 1]
        high_fold_train_n = [i for i in high_fold_train[fold-1] if int(i[1]) == 0]

        # 抽取出各size层验证集并组合
        validation_set = low_fold_train_p[0:int(len(low_fold_train_p) * validation_r)] + \
                         low_fold_train_n[0:int(len(low_fold_train_n) * validation_r)] + \
                         mid_fold_train_p[0:int(len(mid_fold_train_p) * validation_r)] + \
                         mid_fold_train_n[0:int(len(mid_fold_train_n) * validation_r)] + \
                         high_fold_train_p[0:int(len(high_fold_train_p) * validation_r)] + \
                         high_fold_train_n[0:int(len(high_fold_train_n) * validation_r)]

        # 抽取出各size层训练集并组合
        train_set = low_fold_train_p[int(len(low_fold_train_p) * validation_r):] + \
                         low_fold_train_n[int(len(low_fold_train_n) * validation_r):] + \
                         mid_fold_train_p[int(len(mid_fold_train_p) * validation_r):] + \
                         mid_fold_train_n[int(len(mid_fold_train_n) * validation_r):] + \
                         high_fold_train_p[int(len(high_fold_train_p) * validation_r):] + \
                         high_fold_train_n[int(len(high_fold_train_n) * validation_r):]

        test_set = low_fold_test[fold-1]+mid_fold_test[fold-1]+high_fold_test[fold-1]

        return [train_set, validation_set, test_set]