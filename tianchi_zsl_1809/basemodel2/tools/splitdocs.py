"""
@file: splitdocs.py
@author: lishihang 
@software: PyCharm
@time: 2018/09/12
"""

import os,shutil
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r"D:\TianChi\201809ZSL\DatasetA_train_20180813\\")

def copy2docs():

    # 新建文件夹
    def rmrf_mkdir(dirname):
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.mkdir(dirname)

    def move(data,src):

        for k, s in data.iterrows():
            shutil.copy("train/" + s[0], src+ s[1] + "/" + s[0])

    train = pd.read_csv('train.txt', header=None, sep='\t')
    ss = list(set(train.iloc[:, 1]))

    # 创建目录
    rmrf_mkdir("train_img")
    rmrf_mkdir("val_zsl")
    rmrf_mkdir("val_img")

    for s in ss[-10:]:
        temp = train[train.iloc[:, 1]==s]
        print("move to " + "val_zsl/%s/" % s)
        os.makedirs("val_zsl/%s/" % s)
        move(temp,"val_zsl/")

    for s in ss[:-10]:
        temp = train[train.iloc[:, 1]==s]
        print("move to " + "train_img/%s/" % s)
        os.makedirs("train_img/%s/" % s)
        move(temp.iloc[:-10:, :], "train_img/")

        print("move to " + "val_img/%s/" % s)
        os.makedirs("val_img/%s/" % s)
        move(temp.iloc[-10:,:],"val_img/")


if __name__ == '__main__':

    # copy2docs()
    pass

