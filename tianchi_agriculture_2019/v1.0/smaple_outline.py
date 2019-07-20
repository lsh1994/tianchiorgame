"""
@author: LiShiHang
@software: PyCharm
@file: smaple_outline.py
@time: 2019/7/9 19:24
@desc:
"""
import keras
import glob
import numpy as np
import os
import gdal
import cv2
import tqdm
import shutil


def clear_dir(path, model):

    imgs_url = list(glob.glob(path+"0/*.jpg"))

    for i in range(len(imgs_url)):

        imp = imgs_url[i]
        img  = cv2.imread(imp)

        img = cv2.resize(img, (224, 224)) / 255.0

        imgs = np.array([img])
        pred = model.predict(imgs)

        max_index = np.argmax(pred, axis=1)[0]
        max_value = pred[0][max_index]
        print("{}/{} max_index:{} max_value:{:.3f}".format(i,len(imgs_url),max_index,max_value))

        if max_value<0.8:
            os.remove(imp)
            print("confidence low.Remove {}.".format(imp))
        if max_value>0.8 and max_index!=0:
            shutil.move(imp,path+str(max_index))
            print("confidence high in other.Change from{} into {}.".format(imp,str(max_index)))
        print()

    print()

if __name__ == '__main__':


    model = keras.models.load_model("output/0709_1_model.h5")
    model.summary()

    clear_dir("data/val/", model)