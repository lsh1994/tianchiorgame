"""
@file: pred.py
@author: lishihang 
@software: PyCharm
@time: 2018/09/18
"""
import h5py,os
import numpy as np

import keras
from keras import Input, Model
from keras.layers import Dropout, Dense, BatchNormalization,Activation
from keras_preprocessing.image import ImageDataGenerator


import pandas as pd
from sklearn.model_selection import train_test_split
from views import *

dic={'不导电': 0, '其他': 1, '凸粉': 2, '擦花': 3, '桔皮': 4, '横条压凹': 5, '正常': 6, '涂层开裂': 7, '漏底': 8, '碰伤': 9, '脏点': 10, '起坑': 11}
dicsl = {'defect1': 0, 'defect11': 1, 'defect8': 2, 'defect2': 3, 'defect4': 4, 'defect3': 5, 'norm': 6, 'defect9': 7,
           'defect5': 8, 'defect6': 9, 'defect10': 10,'defect7': 11}


def train():
    print(getclassweights())
    X_train, X_val, y_train, y_val = readdata(save_name='train',ts=0.2,nc=12)

    #########################################

    input_tensor = Input(X_train.shape[1:])
    x = Dense(256)(input_tensor)
    x=BatchNormalization()(x)
    x=Activation(activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(12, activation='softmax')(x)
    model = Model(input_tensor, x)

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    his = model.fit(X_train, y_train, batch_size=128, epochs=100, validation_data=(X_val,y_val),
                    verbose=2,class_weight=getclassweights())
    # print(his.history)
    # drawlog(his.history)

    # 验证集上统计
    res=model.predict(X_val)
    wucha(np.argmax(res,axis=1),np.argmax(y_val,axis=1),list(dic.keys()))

    # ########################################
    # 测试集
    X_test,_,_,_ = readdata(save_name='test', ts=0., nc=1)
    y_pred_test = model.predict(X_test, verbose=2)
    degreecf_test = np.max(y_pred_test, axis=1)

    y_p=np.argmax(y_pred_test,axis=1)
    tofile2(y_p,degreecf_test)

def getclassweights():
    path = r"D:\TianChi\defectIdentify\sample\train\\"
    labels = os.listdir(path)
    counts = [len(os.listdir(os.path.join(path, i))) for i in labels]
    return {dic[l]: max(counts)*1.0/ c for l, c in zip(labels,counts)}

def readdata(save_name,ts=0.2,nc=11):
    X_train = []
    for filename in ["gap_InceptionV3.h5"]: #"gap_ResNet50.h5", "gap_Xception.h5"
        with h5py.File("output/%s%s_%s" % (filename[:4], save_name, filename[4:]), 'r') as h:
            X_train.append(np.array(h['data']))
            y_train = np.array(h['label'])
    X_train = np.concatenate(X_train, axis=1)
    y_train = keras.utils.to_categorical(y_train, num_classes=nc)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=ts)
    return X_train, X_val, y_train, y_val


def tofile2(y_pred,degreecf):

    df = pd.read_csv(r"D:\TianChi\defectIdentify\guangdong_round1_submit_sample_20180916.csv",header=None)

    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory(r"D:\TianChi\defectIdentify\sample\test",shuffle=False)

    for i, fname in enumerate(test_generator.filenames):
        index = int(os.path.split(fname)[1].split('.')[0])
        # if degreecf[i]>0.3:
        #     df.iloc[index, 1] = list(dicsl.keys())[y_pred[i]]
        # else:
        #     df.iloc[index, 1]='defect11'
        df.iloc[index, 1] = list(dicsl.keys())[y_pred[i]]

    df.to_csv('output/submit.csv', index=None,header=None,sep=',')
    print(df.head(10))


if __name__ == '__main__':

    train()

    pass