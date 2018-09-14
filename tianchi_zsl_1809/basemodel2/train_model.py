"""
@file: train_vgg.py
@author: lishihang 
@software: PyCharm
@time: 2018/09/08
"""
import keras
from keras import Model, Input
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, \
    GlobalAveragePooling2D, Lambda
from keras.applications import VGG16
import pipeline as pipe
import pipeline2 as pipe2


def MappingModel():

    classes={
        "type":6,
        "color":8,
        "has":4,
        "for":6,
        # "is":6
    }

    basemodel=VGG16(include_top=False,weights="imagenet",input_shape=(64,64,3))
    y=Flatten()(basemodel.output)
    y=Dense(1024,activation='relu')(y)
    # y=Dropout(0.5)(y)
    predense=[Dense(v,activation="softmax",name=k)(y) for k,v in classes.items()]

    my_model = Model(inputs=basemodel.input, outputs=predense)
    trainable=False
    for lay in my_model.layers:
        if(lay.name=="block5_conv1"):
            trainable=True
        lay.trainable=trainable

    my_model.summary()
    keras.utils.plot_model(my_model,to_file='output/mapping_model.png',show_shapes=True)

    my_model.compile(loss='categorical_crossentropy', optimizer="adadelta",metrics=['accuracy'])

    tr_flow = pipe.DataPiple(target=r"D:\TianChi\201809ZSL\DatasetA_train_20180813\train.txt",
                             size=64,impro=True).create_inputs(size=64) #使用数据增强
    checkpoint = ModelCheckpoint(filepath="output/mapping_model.h5", monitor='loss', mode='auto', save_best_only='True')
    # es=keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=50, verbose=2, mode='min')
    tensorboard = TensorBoard(log_dir='output/mapping_vggmodel')
    my_model.fit_generator(tr_flow,steps_per_epoch=32,epochs=2000,verbose=2,callbacks=[tensorboard,checkpoint])

def ImageModel():
    basemodel=VGG16(include_top=False,weights="imagenet",input_shape=(64,64, 3))
    y=Flatten()(basemodel.output)
    y = Dense(1024, activation='relu')(y)
    y = Dense(190, activation='softmax')(y)

    my_model=Model(basemodel.input,y)
    trainable = False
    for lay in my_model.layers:
        if (lay.name == "block5_conv1"):
            trainable = True
        lay.trainable = trainable

    my_model.summary()
    keras.utils.plot_model(my_model,to_file='output/vgg_model.png',show_shapes=True)

    my_model.compile(loss='categorical_crossentropy', optimizer="adadelta",metrics=['accuracy'])


    tr_flow = pipe2.DataPiple(target=r"D:\TianChi\201809ZSL\DatasetA_train_20180813\train.txt", size=64,
                               impro=True).create_inputs(size=64)  # 使用数据增强
    checkpoint = ModelCheckpoint(filepath="output/vgg_model.h5", monitor='acc', mode='auto', save_best_only='True')
    # es = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=2, mode='min') # 提前终止
    tensorboard = TensorBoard(log_dir='output/log_vggmodel')
    my_model.fit_generator(tr_flow, steps_per_epoch=32, epochs=1000, verbose=2,callbacks=[checkpoint,tensorboard])


if __name__ == '__main__':


    MappingModel()

    # ImageModel() # 为了分类训练集和测试集非零样本的类

    pass


