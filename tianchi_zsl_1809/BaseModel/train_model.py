"""
@file: train_vgg.py
@author: lishihang 
@software: PyCharm
@time: 2018/09/08
"""
import keras
from keras import Model, Input
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout
from keras.applications import VGG16,Xception
import BaseModel.pipeline as pipe


def Vgg16Model(size=64):

    basemodel=VGG16(include_top=False,weights="imagenet",input_shape=(size,size, 3))
    model=Model(input=basemodel.input,output=basemodel.output)
    model=Flatten()(model.output)
    model=Dense(1024,activation='relu')(model)
    model=Dense(230,activation="softmax")(model)

    my_model = Model(inputs=basemodel.input, outputs=model)
    trainable=False
    for lay in my_model.layers:
        if(lay.name=="block5_conv1"):
            trainable=True
        lay.trainable=trainable

    my_model.summary()
    keras.utils.plot_model(my_model,to_file='output/vgg_model.png',show_shapes=True)

    my_model.compile(loss='categorical_crossentropy', optimizer="adam",
                     metrics=['accuracy'])
    return my_model

def XceModel(size=71):

    basemodel=Xception(include_top=True,weights="imagenet",input_shape=(size,size, 3))
    model=Model(input=basemodel.input,output=basemodel.layers[-2].output)
    y=Dense(230,activation="softmax",name="classifylayer")(model.output)

    my_model = Model(inputs=model.input, outputs=y)
    trainable=False
    for lay in my_model.layers:
        if(lay.name=="block14_sepconv1"):
            trainable=True
        lay.trainable=trainable

    my_model.summary()
    keras.utils.plot_model(my_model,to_file='output/xce_model.png',show_shapes=True)

    my_model.compile(loss='categorical_crossentropy', optimizer="adam",
                     metrics=['accuracy'])
    return my_model

def deepNet(size=64): #自定义模型

    input = Input(shape=(size, size, 3))

    model = Conv2D(32, (3, 3))(input)
    model = BatchNormalization()(model)
    model = Activation("relu")(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    model = Conv2D(64, (3, 3))(model)
    model = BatchNormalization()(model)
    model = Activation("relu")(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)

    model = Conv2D(64, (3, 3))(model)
    model = BatchNormalization()(model)
    model = Activation("relu")(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.5)(model)
    model = Flatten()(model)

    output = Dense(1024)(model)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)
    output = Dropout(0.5)(output)

    output = Dense(230, activation='softmax')(output)

    my_model = Model(inputs=input, outputs=output)

    my_model.summary()
    keras.utils.plot_model(my_model,to_file='output/dm_model.png',show_shapes=True)

    my_model.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['acc'])
    return my_model


if __name__ == '__main__':
    imgsize=64
    tr_flow = pipe.DataPiple(target="data/train_img.csv", imgsize=imgsize,impro=False).create_inputs(size=64) #不使用数据增强
    va_flow = pipe.DataPiple(target="data/validate_img.csv",imgsize=imgsize,impro=False).create_inputs(size=64)

    my_model=Vgg16Model(size=imgsize)

    checkpoint = ModelCheckpoint(filepath="output/vgg_model.h5", monitor='acc', mode='auto', save_best_only='True')
    tensorboard = TensorBoard(log_dir='output/log_basemodel_vgg')
    my_model.fit_generator(tr_flow,steps_per_epoch=32,epochs=2000,verbose=2,validation_data=va_flow,validation_steps=30,callbacks=[tensorboard,checkpoint])