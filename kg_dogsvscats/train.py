"""
@file: train.py
@author: lishihang 
@software: PyCharm
@time: 2018/09/11
"""

import h5py,os
import numpy as np
from keras import Input, Model
from keras.layers import Dropout, Dense
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
np.random.seed(2018)

os.chdir(r"D:\TianChi\DogvsCat\\")

X_train = []
X_test = []

for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

X_train, y_train = shuffle(X_train, y_train)
# print(X_train[:5])
# print(y_train[:5])
# print(X_train.shape[1:])
#########################################

input_tensor = Input(X_train.shape[1:])
x = Dropout(0.5)(input_tensor)
x = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, x)

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

his=model.fit(X_train, y_train, batch_size=128, epochs=20, validation_split=0.2,verbose=1)
print(his.history)
########################################

y_pred = model.predict(X_test, verbose=2)
y_pred = y_pred.clip(min=0.005, max=0.995)

import pandas as pd

df = pd.read_csv("sample_submission.csv")

gen = ImageDataGenerator()
test_generator = gen.flow_from_directory("test2", (224, 224), shuffle=False,
                                         batch_size=32, class_mode=None)
#
for i, fname in enumerate(test_generator.filenames):
    index = int(os.path.split(fname)[1].split('.')[0])
    df.at[index-1, 'label'] = y_pred[i]

df.to_csv('pred.csv', index=None)
print(df.head(10))