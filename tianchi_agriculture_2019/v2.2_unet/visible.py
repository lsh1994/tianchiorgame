"""
@author: LiShiHang
@software: PyCharm
@file: visible.py
@time: 2019/7/17 14:53
@desc:
"""
import glob
import os
import imutils
import cv2
import numpy as np
from skimage.segmentation import mark_boundaries
from collections import Counter

classes = 4

colors = np.random.randint(0,256,(classes,3))
def label2color(label):

    im = np.zeros(shape=(label.shape[0],label.shape[1],3))
    for i in range(classes):
        im[label==i]=colors[i]
    return im.astype(np.uint8)

for p1 in glob.glob("data/train/imgs/*.png"):

    p2 = p1.replace("imgs","labels")

    img = cv2.imread(p1)
    label = cv2.imread(p2,cv2.IMREAD_GRAYSCALE)

    print(Counter(label.flatten()))

    cv2.imshow("img",imutils.resize(img,height=600))
    cv2.imshow("mask", imutils.resize(label2color(label), height=600))
    cv2.waitKey(0)