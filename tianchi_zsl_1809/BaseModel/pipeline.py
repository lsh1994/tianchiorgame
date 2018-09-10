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

train_attr = pd.read_csv("d:/ZSL_ImageGame/DatasetA_train_20180813/attributes_per_class.txt", sep='\t', header=None)
train_words = pd.read_csv(r"data/word_embeddings.csv", sep='\t', header=None)
labelkeys = sorted(set(train_attr.iloc[:, 0].values.tolist()))

datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,
                                 shear_range=0.2,zoom_range=0.5,horizontal_flip=True,fill_mode='nearest')


class DataPiple:

    def __init__(self,target,imgsize=64,impro=False):
        """

        :param target:
        :param impro: 是否数据增强
        """

        self.target = pd.read_csv(target, header=None, sep='\t').sample(frac=1,random_state=2018)
        self.fea_size=len(self.target)
        self.impro=impro
        self.imgsize=imgsize

    def readOne(self,pos):
        t=self.target.iloc[pos,:]
        im = cv2.imread("d:/ZSL_ImageGame/DatasetA_train_20180813/train/" + t[0])
        im = cv2.resize(im, dsize=(self.imgsize, self.imgsize))
        attr=train_attr[train_attr[0]==t[1]].values[0,1:]
        word=train_words[train_words[0]==t[1]].values[0,1:]
        label=np.zeros(shape=(len(labelkeys)),dtype=np.uint8)
        label[labelkeys.index(t[1])]=1
        return im,attr,word,label



    def readFeather(self,pos,size):
        ims=[]
        attrs=[]
        words=[]
        labels=[]

        for i in range(pos,min(pos+size,self.fea_size)):
            im,attr,word,label=self.readOne(i)
            ims.append(im)
            attrs.append(attr)
            words.append(word)
            labels.append(label)
        ims=np.array(ims)
        # 做数据增强
        if self.impro == True:
            ims=datagen.flow(ims,batch_size=len(ims),shuffle=False).__next__()
        else:
            ims=ims/255.0
        ims = np.array(ims)
        attrs=np.array(attrs)
        words=np.array(words)
        labels=np.array(labels)

        return ims,attrs,words,labels


    def create_inputs(self,size=64):

        while True:
            for i in range(0,self.fea_size,size):
                ims, attrs, words,labels=self.readFeather(i,size)
                # print(ims.shape)
                # print(attrs.shape)
                # print(words.shape)
                yield ims, labels

if __name__ == '__main__':
    # print(labelkeys)

    dp=DataPiple(target=r"D:\ZSL_ImageGame\DatasetA_train_20180813\train.txt",impro=True)
    s=dp.create_inputs(64)
    r,p=s.__next__()

    print(len(r),r.shape,p.shape)

    import matplotlib.pyplot as plt

    plt.imshow(r[2])
    plt.show()

    # print(labelkeys)

