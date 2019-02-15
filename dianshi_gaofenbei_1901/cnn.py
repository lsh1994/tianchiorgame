"""
@author: LiShiHang
@software: PyCharm
@file: cnn.py
@time: 2019/1/28 12:10
@desc: 
"""
import keras
from keras.layers import Input,Dropout,BatchNormalization,Conv2D,Activation,Flatten,Dense
from keras import Sequential,Model
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)  # 使print大量数据不用符号...代替而显示所有

def CNNNet(input_size,bands):
    input = Input(shape=(input_size,input_size,bands))
    o = Conv2D(filters=16,kernel_size=(1,1),strides=(1,1),padding="same")(input)
    o = BatchNormalization()(o)
    o = Activation("relu")(o)
    o=Dropout(0.5)(o)

    o = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation("relu")(o)

    o = Flatten()(o)
    o = Dense(3, activation='softmax')(o)

    model = Model(inputs=input,outputs=o)
    model.summary()
    keras.utils.plot_model(model, to_file='data/model.png', show_shapes=True, show_layer_names=True, rankdir='TB')
    return model

def data_flow(size):

    data = np.load("data/train_raw_{}x{}.npy".format(size,size))
    print(data[:,1].tolist().count(0))
    print(data[:, 1].tolist().count(1))
    print(data[:, 1].tolist().count(2))
    np.random.shuffle(data)
    y= keras.utils.to_categorical(data[:,1],3)
    return np.array([i for i in data[:,0]]),y

if __name__ == '__main__':

    cnn=CNNNet(5,8)
    x,y=data_flow(5)
    print(x.shape,y.shape)
    cnn.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    cnn.fit(x,y,batch_size=32,epochs=20,verbose=2,validation_split=0.2,class_weight=[949,672,751])
    cnn.save("data/model_save.h5")