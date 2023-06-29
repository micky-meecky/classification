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

    # 将nodule的第二个元素提取出来，即label，转换为int，存放到一个新的list中
    label = [int(i[1]) for i in nodules]
    # 排序label，并统计每个label的数量
    label.sort()
    label_count = [label.count(i) for i in range(3)]
    print('label_count', label_count)

    # 提取size的三分点
    sizeall = [int(i[2]) for i in nodules]
    sizeall.sort()  # 按升序排列
    low_mid_thre = sizeall[int(len(sizeall)*4/7)]  # 低中分界点
    mid_high_thre = sizeall[int(len(sizeall)*6/7)]  # 中高分界点

    # 根据size三分位数分为low，mid，high三组

    low_size_list = [i for i in nodules if int(i[2]) < low_mid_thre]
    mid_size_list = [i for i in nodules if int(i[2]) < mid_high_thre and int(i[2]) >= low_mid_thre]
    high_size_list = [i for i in nodules if int(i[2]) >= mid_high_thre]

    # 输出low_size_list的label分布
    low_size_label = [int(i[1]) for i in low_size_list]
    low_size_label.sort()
    print('low_size_label', low_size_label)
    low_size_label_count = [low_size_label.count(i) for i in range(3)]
    print('low_size_label_count', low_size_label_count)

    # 输出mid_size_list的label分布
    mid_size_label = [int(i[1]) for i in mid_size_list]
    mid_size_label.sort()
    print('mid_size_label', mid_size_label)
    mid_size_label_count = [mid_size_label.count(i) for i in range(3)]
    print('mid_size_label_count', mid_size_label_count)

    # 输出high_size_list的label分布
    high_size_label = [int(i[1]) for i in high_size_list]
    high_size_label.sort()
    print('high_size_label', high_size_label)
    high_size_label_count = [high_size_label.count(i) for i in range(3)]
    print('high_size_label_count', high_size_label_count)



    # 将lable划分为三组
    low_label = [int(i[1]) for i in low_size_list]
    mid_label = [int(i[1]) for i in mid_size_list]
    high_label = [int(i[1]) for i in high_size_list]

    # 输出low_label的分布
    low_label.sort()
    print('low_label', low_label)
    low_label_count = [low_label.count(i) for i in range(3)]
    print('low_label_count', low_label_count)

    # 输出mid_label的分布
    mid_label.sort()
    print('mid_label', mid_label)
    mid_label_count = [mid_label.count(i) for i in range(3)]
    print('mid_label_count', mid_label_count)

    # 输出high_label的分布
    high_label.sort()
    print('high_label', high_label)
    high_label_count = [high_label.count(i) for i in range(3)]
    print('high_label_count', high_label_count)


    low_fold_train = []
    low_fold_test = []

    mid_fold_train = []
    mid_fold_test = []

    high_fold_train = []
    high_fold_test = []

    sfolder = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=True)   # 保证每次分折结果相同
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

    if validation is False:  # 不设置验证集，则直接返回
        train_set = low_fold_train[fold-1]+mid_fold_train[fold-1]+high_fold_train[fold-1]
        test_set = low_fold_test[fold-1]+mid_fold_test[fold-1]+high_fold_test[fold-1]

        # 统计训练集和测试集的label分布
        train_label = [int(i[1]) for i in train_set]
        train_label.sort()
        print('train_label', train_label)
        train_label_count = [train_label.count(i) for i in range(3)]
        print('train_label_count', train_label_count)

        test_label = [int(i[1]) for i in test_set]
        test_label.sort()
        print('test_label', test_label)
        test_label_count = [test_label.count(i) for i in range(3)]
        print('test_label_count', test_label_count)

        return [train_set, test_set]

    else:  # 设置验证集合，则从训练集“类别 且 size平衡地”抽取一定数量样本做验证集
        # 分离第fold折各size分层的三类样本，train_n 表示正常类，train_m 表示恶性类，train_b 表示良性类
        low_fold_train_n = [i for i in low_fold_train[fold-1] if int(i[1]) == 2]
        low_fold_train_m = [i for i in low_fold_train[fold-1] if int(i[1]) == 1]
        low_fold_train_b = [i for i in low_fold_train[fold-1] if int(i[1]) == 0]

        mid_fold_train_n = [i for i in mid_fold_train[fold-1] if int(i[1]) == 2]
        mid_fold_train_m = [i for i in mid_fold_train[fold-1] if int(i[1]) == 1]
        mid_fold_train_b = [i for i in mid_fold_train[fold-1] if int(i[1]) == 0]

        high_fold_train_n = [i for i in high_fold_train[fold-1] if int(i[1]) == 2]
        high_fold_train_m = [i for i in high_fold_train[fold-1] if int(i[1]) == 1]
        high_fold_train_b = [i for i in high_fold_train[fold-1] if int(i[1]) == 0]

        # 抽取出各size层验证集并组合
        validation_set = low_fold_train_n[:int(len(low_fold_train_n) * validation_r)] + \
                            low_fold_train_m[:int(len(low_fold_train_m) * validation_r)] + \
                            low_fold_train_b[:int(len(low_fold_train_b) * validation_r)] + \
                            mid_fold_train_n[:int(len(mid_fold_train_n) * validation_r)] + \
                            mid_fold_train_m[:int(len(mid_fold_train_m) * validation_r)] + \
                            mid_fold_train_b[:int(len(mid_fold_train_b) * validation_r)] + \
                            high_fold_train_n[:int(len(high_fold_train_n) * validation_r)] + \
                            high_fold_train_m[:int(len(high_fold_train_m) * validation_r)] + \
                            high_fold_train_b[:int(len(high_fold_train_b) * validation_r)]


        # 抽取出各size层训练集并组合
        train_set = low_fold_train_n[int(len(low_fold_train_n) * validation_r):] + \
                        low_fold_train_m[int(len(low_fold_train_m) * validation_r):] + \
                        low_fold_train_b[int(len(low_fold_train_b) * validation_r):] + \
                        mid_fold_train_n[int(len(mid_fold_train_n) * validation_r):] + \
                        mid_fold_train_m[int(len(mid_fold_train_m) * validation_r):] + \
                        mid_fold_train_b[int(len(mid_fold_train_b) * validation_r):] + \
                        high_fold_train_n[int(len(high_fold_train_n) * validation_r):] + \
                        high_fold_train_m[int(len(high_fold_train_m) * validation_r):] + \
                        high_fold_train_b[int(len(high_fold_train_b) * validation_r):]


        test_set = low_fold_test[fold-1]+mid_fold_test[fold-1]+high_fold_test[fold-1]

        # 获取训练集中各类别样本数量，以及分布情况，并计算各类别样本权重
        train_n = len([i for i in train_set if int(i[1]) == 2])
        train_m = len([i for i in train_set if int(i[1]) == 1])
        train_b = len([i for i in train_set if int(i[1]) == 0])
        train_n_rate = train_n / (train_n + train_m + train_b)
        train_m_rate = train_m / (train_n + train_m + train_b)
        train_b_rate = train_b / (train_n + train_m + train_b)
        train_n_weight = 1 / train_n_rate
        train_m_weight = 1 / train_m_rate
        train_b_weight = 1 / train_b_rate
        # 将权重做归一化处理
        train_n_weight_ = train_n_weight / (train_n_weight + train_m_weight + train_b_weight)
        train_m_weight_ = train_m_weight / (train_n_weight + train_m_weight + train_b_weight)
        train_b_weight_ = train_b_weight / (train_n_weight + train_m_weight + train_b_weight)

        print('train_n: %d, train_m: %d, train_b: %d' % (train_n, train_m, train_b))
        print('train_n_rate: %.4f, train_m_rate: %.4f, train_b_rate: %.4f' % (train_n_rate, train_m_rate, train_b_rate))
        print('train_n_weight: %.4f, train_m_weight: %.4f, train_b_weight: %.4f' % (train_n_weight_, train_m_weight_, train_b_weight_))

        # 获取验证集中各类别样本数量，以及分布情况，并计算各类别样本权重
        validation_n = len([i for i in validation_set if int(i[1]) == 2])
        validation_m = len([i for i in validation_set if int(i[1]) == 1])
        validation_b = len([i for i in validation_set if int(i[1]) == 0])
        validation_n_rate = validation_n / (validation_n + validation_m + validation_b)
        validation_m_rate = validation_m / (validation_n + validation_m + validation_b)
        validation_b_rate = validation_b / (validation_n + validation_m + validation_b)
        validation_n_weight = 1 / validation_n_rate
        validation_m_weight = 1 / validation_m_rate
        validation_b_weight = 1 / validation_b_rate
        # 将权重做归一化处理
        validation_n_weight_ = validation_n_weight / (validation_n_weight + validation_m_weight + validation_b_weight)
        validation_m_weight_ = validation_m_weight / (validation_n_weight + validation_m_weight + validation_b_weight)
        validation_b_weight_ = validation_b_weight / (validation_n_weight + validation_m_weight + validation_b_weight)

        print('validation_n: %d, validation_m: %d, validation_b: %d' % (validation_n, validation_m, validation_b))
        print('validation_n_rate: %.4f, validation_m_rate: %.4f, validation_b_rate: %.4f' % (validation_n_rate, validation_m_rate, validation_b_rate))
        print('validation_n_weight: %.4f, validation_m_weight: %.4f, validation_b_weight: %.4f' % (validation_n_weight_, validation_m_weight_, validation_b_weight_))

        # 获取测试集中各类别样本数量，以及分布情况，并计算各类别样本权重
        test_n = len([i for i in test_set if int(i[1]) == 2])
        test_m = len([i for i in test_set if int(i[1]) == 1])
        test_b = len([i for i in test_set if int(i[1]) == 0])
        test_n_rate = test_n / (test_n + test_m + test_b)
        test_m_rate = test_m / (test_n + test_m + test_b)
        test_b_rate = test_b / (test_n + test_m + test_b)
        test_n_weight = 1 / test_n_rate
        test_m_weight = 1 / test_m_rate
        test_b_weight = 1 / test_b_rate
        # 将权重做归一化处理
        test_n_weight_ = test_n_weight / (test_n_weight + test_m_weight + test_b_weight)
        test_m_weight_ = test_m_weight / (test_n_weight + test_m_weight + test_b_weight)
        test_b_weight_ = test_b_weight / (test_n_weight + test_m_weight + test_b_weight)

        print('test_n: %d, test_m: %d, test_b: %d' % (test_n, test_m, test_b))
        print('test_n_rate: %.4f, test_m_rate: %.4f, test_b_rate: %.4f' % (test_n_rate, test_m_rate, test_b_rate))
        print('test_n_weight: %.4f, test_m_weight: %.4f, test_b_weight: %.4f' % (test_n_weight_, test_m_weight_, test_b_weight_))

        # 计算训练集、验证集、测试集的数量占总数量的比例
        train_rate = (train_n + train_m + train_b) / (train_n + train_m + train_b + validation_n + validation_m + validation_b + test_n + test_m + test_b)
        validation_rate = (validation_n + validation_m + validation_b) / (train_n + train_m + train_b + validation_n + validation_m + validation_b + test_n + test_m + test_b)
        test_rate = (test_n + test_m + test_b) / (train_n + train_m + train_b + validation_n + validation_m + validation_b + test_n + test_m + test_b)

        print('train_rate: %.4f, validation_rate: %.4f, test_rate: %.4f' % (train_rate, validation_rate, test_rate))



        return [train_set, validation_set, test_set]