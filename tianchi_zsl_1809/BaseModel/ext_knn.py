"""
@Time    : 2018/9/7 10:27
@Author  : Lishihang
@File    : ExtKnnNet.py
@Software: PyCharm
@desc: 
"""
from collections import Counter

import pandas as pd
import numpy as np
import cv2
from keras import models
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

labelkeys = ['ZJL1', 'ZJL10', 'ZJL100', 'ZJL101', 'ZJL102', 'ZJL103', 'ZJL104', 'ZJL105', 'ZJL106', 'ZJL107',
                 'ZJL108', 'ZJL109', 'ZJL11', 'ZJL110', 'ZJL111', 'ZJL113', 'ZJL114', 'ZJL115', 'ZJL116', 'ZJL117',
                 'ZJL118', 'ZJL119', 'ZJL12', 'ZJL120', 'ZJL121', 'ZJL122', 'ZJL123', 'ZJL124', 'ZJL125', 'ZJL126',
                 'ZJL127', 'ZJL128', 'ZJL129', 'ZJL13', 'ZJL130', 'ZJL131', 'ZJL132', 'ZJL133', 'ZJL135', 'ZJL137',
                 'ZJL138', 'ZJL139', 'ZJL14', 'ZJL140', 'ZJL141', 'ZJL142', 'ZJL143', 'ZJL144', 'ZJL145', 'ZJL146',
                 'ZJL147', 'ZJL149', 'ZJL15', 'ZJL150', 'ZJL151', 'ZJL152', 'ZJL153', 'ZJL154', 'ZJL156', 'ZJL157',
                 'ZJL158', 'ZJL159', 'ZJL16', 'ZJL160', 'ZJL161', 'ZJL162', 'ZJL163', 'ZJL164', 'ZJL165', 'ZJL166',
                 'ZJL167', 'ZJL168', 'ZJL169', 'ZJL170', 'ZJL171', 'ZJL172', 'ZJL173', 'ZJL174', 'ZJL175', 'ZJL176',
                 'ZJL177', 'ZJL178', 'ZJL179', 'ZJL18', 'ZJL180', 'ZJL181', 'ZJL182', 'ZJL183', 'ZJL184', 'ZJL185',
                 'ZJL186', 'ZJL187', 'ZJL188', 'ZJL189', 'ZJL19', 'ZJL190', 'ZJL191', 'ZJL192', 'ZJL193', 'ZJL194',
                 'ZJL195', 'ZJL2', 'ZJL21', 'ZJL22', 'ZJL23', 'ZJL24', 'ZJL25', 'ZJL26', 'ZJL28', 'ZJL29', 'ZJL3',
                 'ZJL30', 'ZJL31', 'ZJL32', 'ZJL34', 'ZJL35', 'ZJL36', 'ZJL37', 'ZJL38', 'ZJL39', 'ZJL4', 'ZJL40',
                 'ZJL41', 'ZJL42', 'ZJL43', 'ZJL44', 'ZJL45', 'ZJL46', 'ZJL47', 'ZJL48', 'ZJL49', 'ZJL5', 'ZJL50',
                 'ZJL51', 'ZJL52', 'ZJL53', 'ZJL54', 'ZJL55', 'ZJL56', 'ZJL57', 'ZJL58', 'ZJL59', 'ZJL6', 'ZJL60',
                 'ZJL61', 'ZJL62', 'ZJL63', 'ZJL64', 'ZJL65', 'ZJL66', 'ZJL67', 'ZJL68', 'ZJL69', 'ZJL7', 'ZJL70',
                 'ZJL71', 'ZJL72', 'ZJL73', 'ZJL75', 'ZJL76', 'ZJL77', 'ZJL78', 'ZJL79', 'ZJL8', 'ZJL80', 'ZJL81',
                 'ZJL82', 'ZJL83', 'ZJL84', 'ZJL85', 'ZJL86', 'ZJL87', 'ZJL88', 'ZJL89', 'ZJL9', 'ZJL90', 'ZJL91',
                 'ZJL92', 'ZJL93', 'ZJL94', 'ZJL95', 'ZJL96', 'ZJL97', 'ZJL98', 'ZJL99']


def result2submit(labelsrc,imgsrc):
    test = pd.read_csv(labelsrc, sep='\t', header=None)

    # print(test.head())
    test_src = test.iloc[:, 0].values

    md = models.load_model("output/my_model.h5")
    mp = models.load_model("output/mapping.h5")
    md.trainable = False
    mp.trainable=False

    train_attr = pd.read_csv("d:/ZSL_ImageGame/DatasetA_train_20180813/attributes_per_class.txt", sep='\t', header=None)
    attrlabels = set(train_attr.iloc[:, 0])
    attrlabels = attrlabels - set(labelkeys)
    train_attr=train_attr[train_attr.iloc[:,0].isin(attrlabels)]
    # print(train_attr.shape)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_attr.iloc[:, 1:], train_attr.iloc[:, 0])
    # md.summary()

    result = []
    length = len(test_src)
    for i in tqdm(range(length)):
        im = cv2.imread(imgsrc + test_src[i])
        im = cv2.resize(im, dsize=(64, 64))
        # print(im.shape)
        ims = np.array([im/255.0])
        res = md.predict(ims)
        if res[0][np.argmax(res[0])]>=0.5:
            result.append(labelkeys[np.argmax(res[0])])
        else:
            res=mp.predict(ims)
            pred = knn.predict(res)  # 预测的值
            result.append(pred[0])

    result = np.array(result).reshape(-1,1)
    test_src=np.array(test_src).reshape(-1,1)
    result = pd.DataFrame(np.concatenate([test_src,result],axis=1))
    print(result.head())
    result.to_csv("output/submittest.txt", sep='\t', header=None, index=None)

    # true_label = test_src
    #
    # total_counts2 = Counter()
    # for word in true_label:
    #     total_counts2[word] += 1
    # print(total_counts2)

if __name__ == '__main__':

    # result2submit(r"D:\ZSL_ImageGame\DatasetA_test_20180813\DatasetA_test\image.txt",r"D:\ZSL_ImageGame\DatasetA_test_20180813\DatasetA_test\test\\")
    result2submit(r"data/zsl_validate.csv",
                  r"D:\ZSL_ImageGame\DatasetA_train_20180813\train\\")
