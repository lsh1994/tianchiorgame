"""
@author: LiShiHang
@file: train_model.py
@software: PyCharm
@time: 2019/7/6 12:02
@desc:
"""
import keras
from keras.preprocessing.image import ImageDataGenerator
import math

train_datagen = ImageDataGenerator(
    rescale=1./ 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=128,
    class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_directory(
    'data/val',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical')


model = keras.models.load_model("output/0709_1_model.h5")

# 首先，我们只训练顶部的几层（随机初始化的层）
trainable = False
for layer in model.layers:
    if layer.name=="dense_1":
        trainable=True
    layer.trainable = trainable
# keras.utils.plot_model(model, show_shapes=True, to_file='output/model.png')
print(train_generator.class_indices)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc'])
model.summary()

key = "0709_2"
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="output/%s_model.h5" %
    key,
    monitor='val_acc',
    mode='auto',
    save_best_only='True')
tensorboard = keras.callbacks.TensorBoard(log_dir='output/log_%s_model' % key)

model.fit_generator(
    train_generator,
    steps_per_epoch=math.ceil(
        train_generator.samples /
        train_generator.batch_size),
    validation_data=validation_generator,
    epochs=20,
    verbose=1,class_weight = [10, 2, 1, 2],
    callbacks=[
        checkpoint,
        tensorboard])