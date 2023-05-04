import torch
import torch.nn as nn
import torch.nn.functional as F

# SR : Segmentation Result
# GT : Ground Truth
import numpy as np


def get_clsaccuracy(SR, GT):
    acc = 0.
    SR = SR.data.cpu().numpy()
    GT = GT.data.cpu().numpy()
    SR = SR.astype(np.float64)
    GT = GT.astype(np.float64)

    corr = np.sum(SR == GT)
    acc = float(corr) / float(SR.shape[0])

    return acc

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)

    SR = SR.data.cpu().numpy()
    GT = GT.data.cpu().numpy()

    SR = (SR > threshold).astype(np.float64)
    # GT = (GT == (np.max(GT))).astype(np.float64)

    if np.any(GT > 0):
        GT = (GT == np.max(GT)).astype(np.float64)
    else:
        GT = np.zeros_like(GT, dtype=np.float64)

    corr = np.sum(SR == GT)

    acc = float(corr) / float(SR.shape[0])

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)

    SR = SR.data.cpu().numpy()
    GT = GT.data.cpu().numpy()

    SR = (SR > threshold).astype(np.float64)
    # GT = (GT == np.max(GT)).astype(np.float64)

    if np.any(GT > 0):
        GT = (GT == np.max(GT)).astype(np.float64)
    else:
        GT = np.zeros_like(GT, dtype=np.float64)

    # TP : True Positive
    # FN : False Negative
    TP = (((SR == 1.).astype(np.float64) + (GT == 1.).astype(np.float64)) == 2.).astype(np.float64)
    FN = (((SR == 0.).astype(np.float64) + (GT == 1.).astype(np.float64)) == 2.).astype(np.float64)

    corr = np.sum(SR == GT)

    acc = float(corr) / float(SR.shape[0])
    if acc == 1:
        SE = 1  # 因为当acc=1时，说明无论是正样本还是负样本，都被正确分类了，而SE表示的是正样本被正确分类的概率，所以当acc=1时，SE=1
    else:
        SE = float(np.sum(TP)) / (float(np.sum(TP + FN)) + 1e-6)    # 1e-6是为了防止分母为0
    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)

    SR = SR.data.cpu().numpy()
    GT = GT.data.cpu().numpy()

    SR = (SR > threshold).astype(np.float64)
    # GT = (GT == np.max(GT)).astype(np.float64)

    if np.any(GT > 0):
        GT = (GT == np.max(GT)).astype(np.float64)
    else:
        GT = np.zeros_like(GT, dtype=np.float64)

    # TN : True Negative
    # FP : False Positive
    TN = (((SR == 0.).astype(np.float64) + (GT == 0.).astype(np.float64)) == 2.).astype(np.float64)
    FP = (((SR == 1.).astype(np.float64) + (GT == 0.).astype(np.float64)) == 2.).astype(np.float64)

    corr = np.sum(SR == GT)
    acc = float(corr) / float(SR.shape[0])
    if acc == 1:
        SP = 1  # 因为当acc=1时，说明无论是正样本还是负样本，都被正确分类了，而SP表示的是负样本被正确分类的概率，所以当acc=1时，SP=1
    else:
        SP = float(np.sum(TN)) / (float(np.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):

    SR = SR.view(-1)
    GT = GT.view(-1)

    SR = SR.data.cpu().numpy()
    GT = GT.data.cpu().numpy()

    SR = (SR > threshold).astype(np.float64)
    # GT = (GT == (np.max(GT))).astype(np.float64)

    if np.any(GT > 0):
        GT = (GT == np.max(GT)).astype(np.float64)
    else:
        GT = np.zeros_like(GT, dtype=np.float64)

    # TP : True Positive
    # FP : False Positive
    # TP = ((SR == 1) + (GT == 1)) == 2
    # FP = ((SR == 1) + (GT == 0)) == 2

    TP = (((SR == 1.).astype(np.float64) + (GT == 1.).astype(np.float64)) == 2.).astype(np.float64)
    FP = (((SR == 1.).astype(np.float64) + (GT == 0.).astype(np.float64)) == 2.).astype(np.float64)

    corr = np.sum(SR == GT)
    acc = float(corr) / float(SR.shape[0])
    if acc == 1:
        PC = 1  # 因为当acc=1时，说明无论是正样本还是负样本，都被正确分类了，而PC表示的是预测为正的样本中，真正为正的概率，所以当acc=1时，PC=1
    else:
        PC = float(np.sum(TP)) / (float(np.sum(TP + FP)) + 1e-6)
    return PC



def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall F度量
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)
    SR = SR.view(-1)
    GT = GT.view(-1)

    SR = SR.data.cpu().numpy()
    GT = GT.data.cpu().numpy()
    corr = np.sum(SR == GT)
    acc = float(corr) / float(SR.shape[0])
    if acc == 1:
        F1 = 1  # 因为当acc=1时，说明无论是正样本还是负样本，都被正确分类了，而F1表示的是精确度（Precision）和召回率（recall）
                # 是你高我低的关系，所以当acc=1时，F1=1
    else:
        F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity 越大越好

    SR = SR.view(-1)
    GT = GT.view(-1)

    SR = SR.data.cpu().numpy()
    GT = GT.data.cpu().numpy()

    SR = (SR > threshold).astype(np.float64)
    # GT = (GT == (np.max(GT))).astype(np.float64)
    if np.any(GT > 0):  # 如果GT中有正样本，那么GT中的所有元素都置为1
        GT = (GT == np.max(GT)).astype(np.float64)
    else:
        GT = np.zeros_like(GT, dtype=np.float64)

    Inter = np.sum((SR + GT) == 2)
    Union = np.sum((SR + GT) >= 1)

    corr = np.sum(SR == GT)
    acc = float(corr) / float(SR.shape[0])
    if acc == 1:
        JS = 1  # 因为当acc=1时，说明无论是正样本还是负样本，都被正确分类了，而JS表示的是交集和并集的比值，所以当acc=1时，JS=1
    else:
        JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR.view(-1)
    GT = GT.view(-1)

    SR = SR.data.cpu().numpy()
    GT = GT.data.cpu().numpy()

    SR = (SR > threshold).astype(np.float64)
    # GT = (GT == np.max(GT)).astype(np.float64)
    if np.any(GT > 0):
        GT = (GT == np.max(GT)).astype(np.float64)
    else:
        GT = np.zeros_like(GT, dtype=np.float64)

    Inter = np.sum(((SR + GT) == 2).astype(np.float64))

    corr = np.sum(SR == GT)
    acc = float(corr) / float(SR.shape[0])
    if acc == 1:
        DC = 1  # 因为当acc=1时，说明无论是正样本还是负样本，都被正确分类了，而DC表示的是相似度，所以当acc=1时，DC=1
    else:
        DC = float(2 * Inter) / (float(np.sum(SR) + np.sum(GT)) + 1e-6)
    return DC


def get_IOU(SR, GT, threshold=0.5):

    SR = SR.view(-1)
    GT = GT.view(-1)

    SR = SR.data.cpu().numpy()
    GT = GT.data.cpu().numpy()

    SR = (SR > threshold).astype(np.float64)
    # GT = (GT == np.max(GT)).astype(np.float64)

    if np.any(GT > 0):
        GT = (GT == np.max(GT)).astype(np.float64)
    else:
        GT = np.zeros_like(GT, dtype=np.float64)

    # TP : True Positive
    # FP : False Positive
    # FN : False Negative
    TP = (((SR == 1.).astype(np.float64) + (GT == 1.).astype(np.float64)) == 2.).astype(np.float64)
    FP = (((SR == 1.).astype(np.float64) + (GT == 0.).astype(np.float64)) == 2.).astype(np.float64)
    FN = (((SR == 0.).astype(np.float64) + (GT == 1.).astype(np.float64)) == 2.).astype(np.float64)

    corr = np.sum(SR == GT)
    acc = float(corr) / float(SR.shape[0])
    if acc == 1:
        IOU = 1  # 因为当acc=1时，说明无论是正样本还是负样本，都被正确分类了，而IOU表示的是交集和并集的比值，所以当acc=1时，IOU=1
    else:
        IOU = float(np.sum(TP)) / (float(np.sum(TP + FP + FN)) + 1e-6)

    return IOU


def get_all_seg(SR, GT, threshold=0.5):
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)
    F1 = get_F1(SR, GT, threshold=threshold)
    JS = get_JS(SR, GT, threshold=threshold)
    DC = get_DC(SR, GT, threshold=threshold)
    IOU = get_IOU(SR, GT, threshold=threshold)
    Acc = get_accuracy(SR, GT, threshold=threshold)

    return [SE, PC, F1, JS, DC, IOU, Acc]


# 计算分类任务的TP，FP，TN，FN, 输入CR的
def get_TP_FP_TN_FN(predicted, target, class_num):
    """
    Calculate the number of true positive, false positive, true negative, and false negative.
    :param predicted: tensor with predicted labels
    :param target: tensor with ground-truth labels
    :param class_num: number of classes
    :return: TP, FP, TN, FN
    """
    if class_num == 2:
        # For binary classification, convert the predicted values to binary
        predicted[predicted >= 0.5] = 1
        predicted[predicted < 0.5] = 0

    # Calculate TP, FP, TN, FN for each class
    TP = torch.zeros(class_num, dtype=torch.float32)
    FP = torch.zeros(class_num, dtype=torch.float32)
    TN = torch.zeros(class_num, dtype=torch.float32)
    FN = torch.zeros(class_num, dtype=torch.float32)

    for c in range(class_num):
        true_class = (target == c)   # 该类的真实标签
        pred_class = (predicted == c)    # 该类的预测标签
        TP[c] = torch.sum(true_class & pred_class).float()
        FP[c] = torch.sum((~true_class) & pred_class).float()
        TN[c] = torch.sum((~true_class) & (~pred_class)).float()
        FN[c] = torch.sum(true_class & (~pred_class)).float()

    return TP, FP, TN, FN

def get_cls_precision(TP, FP):
    """
    Compute precision score given true positive (TP) and false positive (FP) values.

    Args:
    - TP (list): a list of length n, where n is the number of classes, containing the number of true positive values
    for each class.
    - FP (list): a list of length n, where n is the number of classes, containing the number of false positive values
    for each class.

    Returns:
    - precision (list): a list of length n, where n is the number of classes, containing the precision score for each class.
    """
    precision = []
    for i in range(len(TP)):
        if TP[i] + FP[i] == 0:  # 如果TP+FP=0，说明该类别没有被预测为正样本，此时precision=0
            precision.append(0.0)
        else:
            precision.append(float(TP[i]) / (TP[i] + FP[i]))
    return precision


def get_cls_recall(TP, FN):
    """
    Compute recall score given true positive (TP) and false negative (FN) values.

    Args:
    - TP (list): a list of length n, where n is the number of classes, containing the number of true positive values
    for each class.
    - FN (list): a list of length n, where n is the number of classes, containing the number of false negative values
    for each class.

    Returns:
    - recall (list): a list of length n, where n is the number of classes, containing the recall score for each class.
    """
    recall = []
    for i in range(len(TP)):
        if TP[i] + FN[i] == 0:
            recall.append(0.0)
        else:
            recall.append(float(TP[i]) / (TP[i] + FN[i]))
    return recall


def get_cls_f1_score(TP, FP, FN):
    """
    Compute F1 score given true positive (TP), false positive (FP), and false negative (FN) values.

    Args:
    - TP (list): a list of length n, where n is the number of classes, containing the number of true positive values
    for each class.
    - FP (list): a list of length n, where n is the number of classes, containing the number of false positive values
    for each class.
    - FN (list): a list of length n, where n is the number of classes, containing the number of false negative values
    for each class.

    Returns:
    - f1_score (list): a list of length n, where n is the number of classes, containing the F1 score for each class.
    """
    f1_score = []
    precision = get_cls_precision(TP, FP)
    recall = get_cls_recall(TP, FN)
    for i in range(len(TP)):
        if precision[i] + recall[i] == 0:
            f1_score.append(0.0)
        else:
            f1_score.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))
    return f1_score

def get_cls_precision_w(TP, FP):
    """
    计算二分类任务的精确度（Precision）

    Args:
        TP (List[int]): 二分类任务中的True Positive数，[TP1, TP2]，其中TP1为第0类的TP数，TP2为第1类的TP数
        FP (List[int]): 二分类任务中的False Positive数，[FP1, FP2]，其中FP1为第0类的FP数，FP2为第1类的FP数

    Returns:
        float: 精确度（Precision）的值
    """
    epsilon = 1e-7
    precision = TP[1] / (TP[1] + FP[1] + epsilon)
    return precision


def get_cls_recall_w(TP, FN):
    """
    计算二分类任务的召回率（Recall）

    Args:
        TP (List[int]): 二分类任务中的True Positive数，[TP1, TP2]，其中TP1为第0类的TP数，TP2为第1类的TP数
        FN (List[int]): 二分类任务中的False Negative数，[FN1, FN2]，其中FN1为第0类的FN数，FN2为第1类的FN数

    Returns:
        float: 召回率（Recall）的值
    """
    epsilon = 1e-7
    recall = TP[1] / (TP[1] + FN[1] + epsilon)
    return recall


def get_cls_f1_score_w(TP, FP, FN):
    """
    计算二分类任务的F1-score

    Args:
        TP (List[int]): 二分类任务中的True Positive数，[TP1, TP2]，其中TP1为第0类的TP数，TP2为第1类的TP数
        FP (List[int]): 二分类任务中的False Positive数，[FP1, FP2]，其中FP1为第0类的FP数，FP2为第1类的FP数
        FN (List[int]): 二分类任务中的False Negative数，[FN1, FN2]，其中FN1为第0类的FN数，FN2为第1类的FN数

    Returns:
        float: F1-score的值
    """
    epsilon = 1e-7
    precision = TP[1] / (TP[1] + FP[1] + epsilon)
    recall = TP[1] / (TP[1] + FN[1] + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1_score





