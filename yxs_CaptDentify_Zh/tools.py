import os
import glob
from skimage import io
import cv2
import numpy as np
import tqdm
import pandas as pd
import imutils
import string
import matplotlib.pyplot as plt
import random
import collections
import shutil
from scipy import stats


def cap_show(csv="data/train.csv", imgdir="data/train/"):

    data = pd.read_csv(csv)
    print(data.head())

    labels = "".join(data.label.tolist())
    print(set(labels)) # 唯一值，调试可得全部3500
    print(len(labels), len(labels) / data.shape[0])  # 平均长度
    print(len(dict(collections.Counter(labels))))
    print(sorted(dict(collections.Counter(labels)).items(), key=lambda x: x[0]))  # 各字符数量
    print(collections.Counter([len(i) for i in data.label.tolist()]))  # 长度分布

    for i, d in data.iterrows():
        # d.filename = "4.jpg"
        img = io.imread(imgdir + d.filename, as_gray=False)
        print(i, d.filename,d.label, img.shape)
        io.imshow(img)
        io.show()

# 拆分训练集、测试集


def split_data():

    data = pd.read_csv("data/train.csv")
    split = int(len(data) * 0.9)
    data = data.sample(frac=1, random_state=2020)  # 打乱
    data[:split].to_csv("data/train_train.csv", index=False)
    data[split:].to_csv("data/train_val.csv", index=False)


# 结果投票

def vote():
    path = list(glob.glob("submit_all/*.csv"))
    r = pd.read_csv(path[0], header=None)
    for i in path[1:]:
        r = pd.merge(r, pd.read_csv(i, header=None), on=0)
    print(r.head())
    r = r.to_numpy()

    res = []
    for i in tqdm.tqdm(range(len(r))):
        res.append(stats.mode(r[i, 1:])[0])
    r = np.concatenate((r[:, 0].reshape(-1, 1), np.array(res).reshape(-1,1)), axis=1)
    r = pd.DataFrame(r, columns=["filename", "label"])
    r.to_csv(f"submit_all/vote_submit.csv", index=False, header=False)


if __name__ == '__main__':


    # cap_show()

    # split_data()
    vote()
    pass
