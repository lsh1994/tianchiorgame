"""
@file: docsdeal.py
@author: lishihang 
@software: PyCharm
@time: 2018/09/19
"""

import os
import shutil

from pyecharts import Line,Bar
import keras
from keras_preprocessing import image
from keras import preprocessing
from keras_preprocessing.image import ImageDataGenerator


def copydoc():
    """
    拷贝其他文件夹大于等于10个的样本文件夹，并删除其他文件夹
    :return:
    """
    path=r"D:\TianChi\defectIdentify\sample\train\\其他\\"
    d=os.listdir(path)
    d_len=[len(os.listdir(os.path.join(path,i))) for i in d]

    for i,j in zip(d,d_len):

        if j>=10:
            print(i, j)
            if not os.path.exists(r"D:\TianChi\defectIdentify\sample\train\\"+i):
                shutil.copytree(os.path.join(path,i),r"D:\TianChi\defectIdentify\sample\train\\"+i)
    shutil.rmtree(path)

def show():
    """
    显示文件夹子文件夹下图片统计表
    :return:
    """
    path=r"D:\TianChi\defectIdentify\sample\train\\"
    d=os.listdir(path)
    d_len=[len(os.listdir(os.path.join(path,i))) for i in d]

    line = Bar(path)
    line.add(path, d, d_len, mark_point=["average", "max", "min"],xaxis_rotate=40)
    line.render(path="../output/temp.html")

def data_aug():
    """
    数据增强，数量91~100
    :return:
    """
    path=r"D:\TianChi\defectIdentify\sample\train\\"
    image_datagen = ImageDataGenerator(rotation_range=360,shear_range=0.2,zoom_range=0.2,
                                       cval=0,fill_mode='constant',horizontal_flip=True,vertical_flip=True)
    for d in os.listdir(path):
        print("dealing:",d)
        td=r"D:\TianChi\defectIdentify\sample\train_gen\\"+d
        if os.path.exists(td):
            shutil.rmtree(td)
        os.makedirs(td)
        image_generator = image_datagen.flow_from_directory(
            path,target_size=(800,800),classes=[d],
            class_mode=None,batch_size=1,shuffle=True,
            save_to_dir=td)
        for _ in range(200):
            image_generator.next()

def movefiles():
    """
    递归移动文件夹中的所有文件到指定目录
    :return:
    """
    src=r"D:\TianChi\defectIdentify\其他\\"
    dsc=r"D:\TianChi\defectIdentify\others\valother\\"
    for root, dirs, files in os.walk(src):
        for f in files:
            shutil.copy(os.path.join(root, f), os.path.join(dsc, f))

if __name__ == '__main__':

    show()
    # copydoc()
    # data_aug()
    # movefiles()
    pass
