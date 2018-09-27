"""
@file: code1.py
@author: lishihang 
@software: PyCharm
@time: 2018/09/21
"""
import math

import keras
import numpy as np
import pandas as pd
import os
import random
import shutil
from keras.applications import Xception
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,TensorBoard
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

path=r"D:\TianChi\kgPlantSeedlings\sample\\"

def split_validation_set():
    os.mkdir(path+'dev')
    for category in CATEGORIES:
        os.mkdir(path+'dev/' + category)
        name = os.listdir(path+'train/' + category)
        random.shuffle(name)
        todev = name[:int(len(name) * .2)]
        for file in todev:
            shutil.move(os.path.join(path+'train', category, file), os.path.join(path+'dev', category))

def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def train():
    # data generator
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=50,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

    train_generator = train_datagen.flow_from_directory(
        path+'train',
        target_size=(299, 299),
        batch_size=16,
        class_mode='categorical',
        shuffle=True)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    val_generator = val_datagen.flow_from_directory(
        path+'dev',
        target_size=(299, 299),
        batch_size=16,
        class_mode='categorical',
        shuffle=True)
    pre=pretrain_dense_layer(train_generator,val_generator)
    train_whole_model(*pre)



def pretrain_dense_layer(train_generator,val_generator):
    tensorboard = TensorBoard(r'output/logs')

    basic_model = Xception(include_top=False, weights='imagenet', pooling='avg')

    for layer in basic_model.layers:
        layer.trainable = False

    input_tensor = basic_model.input
    # build top
    x = basic_model.output
    x = Dropout(.5)(x)
    x = Dense(len(CATEGORIES), activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=x)
    model.compile(optimizer=RMSprop(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(train_generator, epochs=40,
                        validation_data=val_generator,
                        callbacks=[tensorboard],
                        workers=1,
                        verbose=1)
    return model,train_generator,val_generator,tensorboard

def train_whole_model(model,train_generator,val_generator,tensorboard):
    for layer in model.layers:
        layer.W_regularizer = l2(1e-2)
        layer.trainable = True

    model.compile(optimizer=RMSprop(lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])

    # call backs
    checkpointer = ModelCheckpoint(filepath='output/weights_xception.h5', verbose=1,
                                   save_best_only=True)

    lr = LearningRateScheduler(lr_schedule)

    # train dense layer
    model.fit_generator(train_generator,
                        steps_per_epoch=400,
                        epochs=150,
                        validation_data=val_generator,
                        callbacks=[checkpointer, tensorboard, lr],
                        initial_epoch=40,
                        workers=1,
                        verbose=1)

def predict():
    model=keras.models.load_model("output/weights_xception.h5")

    class_indices = {'Black-grass': 0, 'Charlock': 1, 'Cleavers': 2, 'Common Chickweed': 3, 'Common wheat': 4,
                     'Fat Hen': 5, 'Loose Silky-bent': 6, 'Maize': 7, 'Scentless Mayweed': 8, 'Shepherds Purse': 9,
                     'Small-flowered Cranesbill': 10, 'Sugar beet': 11}

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        path + 'test',
        target_size=(299, 299),
        batch_size=16,
        class_mode=None,
        shuffle=False)

    test=model.predict_generator(test_generator, steps=math.ceil(test_generator.samples*1./test_generator.batch_size), verbose=1)
    res=[list(class_indices.keys())[i] for i in np.argmax(test,axis=1)]
    imgs=[os.path.split(i)[1] for i in test_generator.filenames]

    res=pd.DataFrame(data={"file":imgs,"species":res})
    print(res.head())
    res.to_csv("output/submit.csv",index=None,sep=',')


if __name__ == '__main__':

    # split_validation_set()

    # train()

    predict()
    pass