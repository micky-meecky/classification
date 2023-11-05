#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: analysis.py
@datatime: 11/5/2023 1:24 PM
"""
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def ROC_Curve():
    # 假设你有一个模型名称和对应文件路径的字典
    model_files = {
        'N-Unet': 'model_predictions_nunet.csv',
        'ANUnet': 'model_predictions_ANunet.csv',
        'AGUNet': 'model_predictions_AGunet.csv',
        'M-Unet': 'model_predictions_Munet.csv',
        'ResUnet': 'model_predictions_resunet.csv',
        'ViT-Unet': 'model_predictions_ViTUnet.csv',
        'Unet': 'model_predictions_unet.csv',
        # ... 可以添加更多模型
    }

    # 设置线条样式，确保有足够的样式供所有模型使用
    line_styles = ['-o', '.-', '--', '-.', '-..', '-v', ':', '-D', '-']

    plt.figure()
    for (model_name, file_path), line_style in zip(model_files.items(), line_styles):
        # 读取数据
        data = pd.read_csv(file_path)
        true_labels = data['TrueLabel']
        predicted_probabilities = data['PredictedProbability']

        # 计算ROC曲线的各个点
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities, pos_label=1)

        # 计算AUC值
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线
        plt.plot(fpr, tpr, line_style,
                 label=f'{model_name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def Confusion_matrix():
    import numpy as np
    import seaborn as sns

    # 设定混淆矩阵的值
    tp = 126
    fp = 5
    tn = 23
    fn = 3

    # 创建混淆矩阵数组
    confusion_matrix = np.array([[tp, fn], [fp, tn]])

    # 使用Seaborn的heatmap函数来画混淆矩阵
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Positive", "Predicted Negative"],
                yticklabels=["Actual Positive", "Actual Negative"])

    # 设置标题和坐标轴标签
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()


if __name__ == '__main__':
    # ROC_Curve()
    Confusion_matrix()
