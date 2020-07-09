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


def cap_show(csv="data/test.csv", imgdir="data/test/"):

    data = pd.read_csv(csv)
    print(data.head())

    labels = "".join(data.label.tolist())
    print(len(labels), len(labels) / data.shape[0])  # 平均长度
    print(sorted(dict(collections.Counter(labels)).items(), key=lambda x: x[0]))  # 各字符数量
    print(collections.Counter([len(i) for i in data.label.tolist()]))  # 长度分布
    # 每个字母有多少训练数据出现
    key = string.digits + string.ascii_letters
    r = {}
    for k in tqdm.tqdm(key):
        m = 0
        for i, d in data.iterrows():
            if k in d.label:
                m += 1
        r[k] = m
    print(r)

    for i, d in data.iterrows():

        img = cv2.imread(imgdir + d.filename, cv2.IMREAD_COLOR)
        print(i, d.label, img.shape)
        cv2.imshow("1", img)
        cv2.waitKey(0)

# 合并train2到train1


def reflush_train2():
    res = []
    for i in tqdm.tqdm(os.listdir("data/train2/")):
        label = i.split(".")[0]
        shutil.copy("data/train2/" + i, "data/train/")  # 重复会被替换
        res.append([i, label])
    shutil.rmtree("data/train2")

    old = pd.read_csv("data/train.csv").values.tolist()

    new = pd.DataFrame(data=np.array(res + old))
    new.to_csv("data/train.csv", index=False, header=["filename", "label"])

# 拆分训练集、测试集


def split_data():

    data = pd.read_csv("data/train.csv")
    split = int(len(data) * 0.85)
    data = data.sample(frac=1, random_state=2020)  # 打乱
    data[:split].to_csv("data/train_train.csv", index=False)
    data[split:].to_csv("data/train_val.csv", index=False)

# 重复难例


def train_repeat(csv="data/train_train.csv", times=2):
    data = pd.read_csv(csv)
    print(data.head(), data.shape)

    ka = list("0oO")

    res = []
    count = 0
    for i, d in data.iterrows():
        if len(set(list(d.label)) & set(ka)) != 0:
            count += 1
            res.append(d.tolist())
    print(count)
    print(res)
    res = pd.DataFrame(res * times, columns=["filename", "label"])

    data = pd.concat((data, res), axis=0)
    print(data.shape)
    data.to_csv("data/train_repeat.csv", index=False)

# 伪标签


def pseudo_labeling(train_csv, submit_csv,frac):
    train = pd.read_csv(train_csv)
    submit = pd.read_csv(submit_csv, header=None)

    res = []
    for i in tqdm.tqdm(os.listdir("data/test/")):
        label = submit[submit.iloc[:, 0] == int(i.split(".")[0])].values[0, 1]
        newfile = "test_" + i
        shutil.copy("data/test/" + i, "data/train/" + newfile)  # 重复会被替换
        res.append([newfile, label])
    res = pd.DataFrame(res, columns=["filename", "label"])

    # print(res.head(),res.shaoe)
    res = res.sample(frac=frac,random_state=2020)  # 伪标签数据采样
    res = pd.concat((train, res), axis=0)
    res.to_csv("data/train_train_pseudo.csv", index=False)

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
    r = np.concatenate((r[:, 0].reshape(-1, 1), np.array(res).reshape(-1, 1)), axis=1)
    r = pd.DataFrame(r, columns=["filename", "label"])
    r.to_csv("submit_all/vote_submit.csv", index=False, header=False)


if __name__ == '__main__':

    # cap_show()
    # reflush_train2()
    # split_data()
    # train_repeat()
    vote()
    # pseudo_labeling("data/train_train.csv", "submit_ass/vote_submit.csv",0.5)
    pass
