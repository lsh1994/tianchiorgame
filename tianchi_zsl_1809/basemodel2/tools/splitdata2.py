"""
@Time    : 2018/9/5 12:33
@Author  : Lishihang
@File    : splitdata2.py.py
@Software: PyCharm
@desc: 
"""


import numpy as np
import pandas as pd

src="d:/ZSL_ImageGame/DatasetA_train_20180813/"
train=pd.read_csv(src+'train.txt',header=None,sep='\t').sample(frac=1)
# print(train.head())
print("train.txt数量",train.shape)

s= ["ZJL"+str(i) for i in range(196,201)]
# print(s)

zsl_validate=train[train.iloc[:,1].isin(s)].sample(frac=1,random_state=2018)
# print(validate.head())
print("零样本测试集",zsl_validate.shape)

traindata=train[~train.iloc[:,1].isin(s)].sample(frac=1,random_state=2018)
# print("",traindata.shape)

train_img=train.iloc[:-2000,:]
print("图片训练集",train_img.shape)
validate_img=train.iloc[-2000:,:]
print("图片测试集",validate_img.shape)


#
# zsl_validate.to_csv("../data/zsl_validate.csv", sep="\t", header=None, index=None)
train_img.to_csv("../data/train_img.csv", sep="\t", header=None, index=None)
validate_img.to_csv("../data/validate_img.csv", sep="\t", header=None, index=None)