import torch
import torch.nn as nn
import torch.nn.functional as F

# SR : Segmentation Result
# GT : Ground Truth
import numpy as np


def get_clsaccuracy(SR, GT):
    acc = 0
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
    if np.any(GT > 0):
        GT = (GT == np.max(GT)).astype(np.float64)
    else:
        GT = np.zeros_like(GT, dtype=np.float64)

    Inter = torch.sum((SR + GT) == 2)
    Union = torch.sum((SR + GT) >= 1)

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

