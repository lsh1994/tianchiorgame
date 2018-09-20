"""
@file: pred.py
@author: lishihang 
@software: PyCharm
@time: 2018/09/18
"""
import h5py,os
import numpy as np
from keras import Input, Model
from keras.layers import Dropout, Dense, BatchNormalization,Activation
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from pyecharts import Line
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(2018)
os.chdir(r"D:\TianChi\defectIdentify\sample\\")

dic = {'defect1': 0, 'defect11': 1, 'defect8': 2, 'defect2': 3, 'defect4': 4, 'defect3': 5, 'norm': 6, 'defect9': 7,
           'defect5': 8, 'defect6': 9, 'defect10': 10,'defect7': 11}


def train():
    X_train, y_train, X_val, y_val,X_test=readdata()

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

    his = model.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_val,y_val),verbose=2,class_weight='auto')
    # print(his.history)
    drawlog(his)

    res=model.predict(X_val)
    wucha(np.argmax(res,axis=1),np.argmax(y_val,axis=1),list(dic.keys()))

    # ########################################

    y_pred = model.predict(X_test, verbose=2)
    y_pred=np.argmax(y_pred,axis=1)
    tofile2(y_pred)

def wucha(fl,tl,label):
    print(classification_report(tl, fl))
    print("acc:",accuracy_score(tl,fl))
    mat = confusion_matrix(tl, fl)
    sns.heatmap(mat, annot=True, square=True, fmt="d", xticklabels=label, yticklabels=label)
    plt.show()


def readdata():

    X_train = []
    X_test = []

    for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:
        with h5py.File(filename, 'r') as h:
            X_train.append(np.array(h['train']))
            X_test.append(np.array(h['test']))
            y_train = np.array(h['label'])

    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)

    y_train = keras.utils.to_categorical(y_train)

    X_train,  X_val, y_train,y_val =train_test_split(X_train, y_train,test_size=0.4)
    # print(X_train[:5])
    # print(y_train[:5])
    # print(X_train.shape[1:])

    return X_train,y_train,X_val,y_val,X_test

def tofile2(y_pred):


    df = pd.read_csv("../guangdong_round1_submit_sample_20180916.csv",header=None)

    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory("test", (224, 224), shuffle=False,batch_size=32, class_mode=None)

    for i, fname in enumerate(test_generator.filenames):
        index = int(os.path.split(fname)[1].split('.')[0])
        df.iloc[index, 1] = list(dic.keys())[y_pred[i]]

    df.to_csv('submit.csv', index=None,header=None,sep=',')
    print(df.head(10))

def drawlog(his):
    line = Line("log")
    for label, value in his.history.items():
        line.add(label, [i for i in range(len(value))], value)
    line.render(path="temp.html")

if __name__ == '__main__':

    train()
