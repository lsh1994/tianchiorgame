"""
@author: LiShiHang
@software: PyCharm
@file: model.py
@time: 2019/2/20 11:01
@desc:
"""
import keras
from keras.layers import Input, Dropout, BatchNormalization, Conv2D, Activation, Flatten, Dense, GlobalMaxPooling2D, MaxPool2D
from keras import Model
import numpy as np
from dataflow import TrainFlow
import math

np.set_printoptions(threshold=np.inf)  # 使print大量数据不用符号...代替而显示所有


def get_model(size):
    input = Input(shape=(size, size, 3))

    o = Conv2D(
        filters=32, kernel_size=3, strides=1)(input)
    o = BatchNormalization()(o)
    o = Activation(activation="relu")(o)
    o = MaxPool2D()(o)

    o = Conv2D(
        filters=64, kernel_size=3, strides=1)(o)
    o = BatchNormalization()(o)
    o = Activation(activation="relu")(o)
    o = MaxPool2D()(o)

    o = Flatten()(o)
    o = Dropout(0.5)(o)
    o = Dense(4, activation="softmax")(o)

    mymodel = Model(inputs=input, outputs=o)
    mymodel.summary()

    keras.utils.plot_model(
        mymodel,
        to_file='output/model.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB')
    return mymodel


def train():
    train_batch_size = 256
    size = 25
    label = "data/train_enhance.txt"
    tif = r"E:\机器学习竞赛\baidu_dianshi\rgb_data.tif"
    bands = [1, 2, 3]

    flow = TrainFlow(
        label=label,
        tif=tif,
        size=size,
        bands=bands)

    cnn = get_model(size)
    cnn.compile(

        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy'])
    cnn.fit_generator(flow.get_batch(train_batch_size, enhance=True),
                      steps_per_epoch=math.ceil(flow.label_length / train_batch_size), epochs=20, verbose=2,
                      class_weight=[1000 / 949, 1000 / 751, 1000 / 672, 1000 / 3000]
    )

    cnn.save("output/model_save.h5")

    validate()



def validate():
    train_batch_size = 256
    size = 25
    label = "data/train_enhance.txt"
    tif = r"E:\机器学习竞赛\baidu_dianshi\rgb_data.tif"
    bands = [1, 2, 3]

    cnn = keras.models.load_model("output/model_save.h5")
    cnn.summary()
    # 评估
    flow2 = TrainFlow(
        label=label,
        tif=tif,
        size=size,
        bands=bands)
    fb = flow2.get_batch(train_batch_size)
    predict_label = []
    true_label = []
    for i in range(math.ceil(flow2.label_length / train_batch_size)):
        x, y = fb.__next__()
        predict_label += np.argmax(cnn.predict_on_batch(x), axis=1).tolist()
        true_label += np.argmax(y, axis=1).tolist()
    from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score, confusion_matrix
    print(confusion_matrix(true_label, predict_label))
    print(classification_report(true_label, predict_label))
    print("kappa: ", cohen_kappa_score(true_label, predict_label))
    print("accuracy: ", accuracy_score(true_label, predict_label))

if __name__ == '__main__':

    # train()
    validate()
