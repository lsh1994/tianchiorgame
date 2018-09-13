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



def result2submit(labelsrc,imgsrc):
    target = pd.read_csv(r"D:\TianChi\201809ZSL\DatasetA_train_20180813\train.txt", header=None, sep='\t')
    labelkeys190 = sorted(set(target.iloc[:, 1].values.tolist()))

    train_attr = pd.read_csv(r"data\attributes_per_class_norm.csv", sep='\t', header=None)
    labelkeys230 = set(train_attr.iloc[:, 0].values.tolist())

    labelkeys40 = labelkeys230 - set(labelkeys190)


    test = pd.read_csv(labelsrc, sep='\t', header=None)
    # print(test.head())
    test_src = test.iloc[:, 0].values

    vgg_model = models.load_model("output/vgg_model.h5")
    image_model = models.load_model("output/mapping_model.h5")
    vgg_model.trainable = False
    image_model.trainable=False

    train_knn = train_attr[ train_attr.iloc[:,0].isin(labelkeys40)]
    # print(train_attr.shape)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_knn.iloc[:, 1:], train_knn.iloc[:, 0])
    # md.summary()

    result = []
    length = len(test_src)
    for i in tqdm(range(length)):
        im = cv2.imread(imgsrc + test_src[i])
        im = cv2.resize(im, dsize=(64, 64))
        # print(im.shape)
        ims = np.array([im/255.0])
        res = vgg_model.predict(ims)
        if res[0][np.argmax(res[0])]>=0.5:
            result.append(labelkeys190[np.argmax(res[0])])
        else:
            res=image_model.predict(ims)
            pred = knn.predict([np.hstack((res[0][0],res[1][0],res[2][0],res[3][0]))])  # 预测的值
            result.append(pred[0])

    result = np.array(result).reshape(-1,1)
    test_src=np.array(test_src).reshape(-1,1)
    result = pd.DataFrame(np.concatenate([test_src,result],axis=1))
    result=pd.DataFrame(result)
    result.to_csv("output/submittest.txt", sep='\t', header=None, index=None)

if __name__ == '__main__':

    result2submit(r"D:\TianChi\201809ZSL\DatasetA_test_20180813\DatasetA_test\image.txt",r"D:\TianChi\201809ZSL\DatasetA_test_20180813\DatasetA_test\test\\")
    # result2submit(r"data/zsl_validate.csv",r"D:\ZSL_ImageGame\DatasetA_train_20180813\train\\")
