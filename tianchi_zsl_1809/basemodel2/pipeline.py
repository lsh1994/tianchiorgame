"""
@file: pipeline.py
@author: lishihang 
@software: PyCharm
@time: 2018/09/08
"""
import keras
import pandas as pd
import numpy as np
import cv2
from keras_preprocessing.image import ImageDataGenerator

train_attr = pd.read_csv(r"data/attributes_per_class_norm.csv", sep='\t', header=None)

datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,
                                 shear_range=0.1,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')


class DataPiple:

    def __init__(self,target,size=64,impro=False):
        """

        :param target:
        :param impro: 是否数据增强
        """

        self.target = pd.read_csv(target, header=None, sep='\t').sample(frac=1)
        self.fea_size=len(self.target)
        self.impro=impro
        self.size=size

    def readOne(self,pos):
        t=self.target.iloc[pos,:]
        im = cv2.imread(r"D:\TianChi\201809ZSL\DatasetA_train_20180813\train\\" + t[0])
        im = cv2.resize(im, dsize=(self.size, self.size))
        attr=train_attr[train_attr[0]==t[1]].values[0,1:]
        return im,attr



    def readFeather(self,pos,size):
        ims=[]
        attrs=[]

        for i in range(pos,min(pos+size,self.fea_size)):
            im,attr=self.readOne(i)
            ims.append(im)
            attrs.append(attr)
        ims=np.array(ims)
        # 做数据增强
        if self.impro == True:
            ims=datagen.flow(ims,batch_size=len(ims),shuffle=False).__next__()
        else:
            ims=ims/255.0
        ims = np.array(ims)
        attrs=np.array(attrs)

        return ims,attrs


    def create_inputs(self,size=64):

        while True:
            for i in range(0,self.fea_size,size):
                ims, attrs=self.readFeather(i,size)
                # print(ims.shape)
                # print(attrs.shape)
                yield ims, [attrs[:,0:6],attrs[:,6:14],attrs[:,14:18],attrs[:,18:24]]

if __name__ == '__main__':
    # print(labelkeys)

    dp=DataPiple(target=r"D:\TianChi\201809ZSL\DatasetA_train_20180813\train.txt",impro=False,size=64)
    s=dp.create_inputs(64)
    r,p=s.__next__()

    print(len(r),r.shape,p)

    import matplotlib.pyplot as plt

    plt.imshow(r[2])
    plt.show()

    # print(labelkeys)

