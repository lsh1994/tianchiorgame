"""
@author: LiShiHang
@software: PyCharm
@file: train.py
@time: 2019/7/17 16:01
@desc:
"""
from segmentation_models import Unet
from keras.layers import Input, Conv2D
from keras.models import Model
from segmentation_models.losses import cce_dice_loss
from segmentation_models.metrics import iou_score, f1_score
from keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau,EarlyStopping
import keras
import utils
import math
import glob

train_batch_size = 64

##################################################
model = Unet(
    backbone_name='densenet121',
    encoder_weights='imagenet',
    input_shape=(384, 384, 3),
    encoder_freeze=True,
    classes=4,
    activation="softmax")
model.compile('adam', loss=cce_dice_loss, metrics=["acc",iou_score])

model.summary()
# keras.utils.plot_model(model, "output/umodel.png", show_shapes=True)

##################################################

G = utils.get_flow(train_batch_size, 1152, argument=True)

key = "tc_arg"
checkpoint = ModelCheckpoint(
    filepath="output/%s_model.h5" % key,
    monitor='iou_score',
    mode='max', verbose=1,
    save_best_only='True')
reduce_lr = ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,min_lr=1e-7,
    patience=10,
    verbose=1,
    mode='auto')
early_stopping = EarlyStopping(patience=30, verbose=1,monitor="iou_score")
ton = keras.callbacks.TerminateOnNaN()

len_s = len(list(glob.glob("data/train/imgs/*.png")))
model.fit_generator(generator=G,
                steps_per_epoch=math.ceil(len_s / train_batch_size),
                epochs=1000, callbacks=[checkpoint, ton], shuffle=True,
                verbose=1)
