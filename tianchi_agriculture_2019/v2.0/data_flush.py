"""
@author: LiShiHang
@file: data_flush.py
@software: PyCharm
@time: 2019/7/6 10:59
@desc:
"""
import pandas as pd
import numpy as np
import os
import shutil
from collections import Counter
from sklearn.model_selection import train_test_split
import tqdm
import sys


def min_mkdir(s):

    if os.path.exists(s):
        shutil.rmtree(s)
    os.mkdir(s)


def data_flush(label2txt, classes):

    labels = pd.read_csv(label2txt, header=None, sep="\t").to_numpy()

    am = np.argmax(labels[:, 1:], axis=1)  # 最大值索引
    m = np.max(labels[:, 1:], axis=1)  # 最大值
    labels = np.concatenate(
        (labels, am.reshape(-1, 1), m.reshape(-1, 1)), axis=1)

    print("Before screen：", labels.shape)
    labels = labels[labels[:, -1] > 0.75]
    print("After screen：", labels.shape)
    print()

    print("Before split：", Counter(labels[:, -2]))
    X_train, X_test, y_train, y_test = train_test_split(labels, labels[:, -2], test_size=0.3, stratify=labels[:, -2],
                                                        random_state=2019)
    print("After split：", Counter(y_train), Counter(y_test))
    print()

    for i in tqdm.tqdm(range(len(X_train))):
        shutil.copy(X_train[i, 0], "data/train/" + str(X_train[i, -2]))

    for i in tqdm.tqdm(range(len(X_test))):
        shutil.copy(X_test[i, 0], "data/val/" + str(X_test[i, -2]))

    print("finish.")


if __name__ == '__main__':

    min_mkdir("data/")
    min_mkdir("data/train")
    min_mkdir("data/val")
    for i in range(4):
        min_mkdir("data/train/" + str(i))
        min_mkdir("data/val/" + str(i))
    min_mkdir("output")
    min_mkdir("output/pred")

    data_flush("../data/train/labels2txt.txt", classes=4)
