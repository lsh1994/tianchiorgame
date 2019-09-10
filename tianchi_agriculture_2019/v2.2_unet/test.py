"""
@author: LiShiHang
@software: PyCharm
@file: test.py
@time: 2019/7/17 19:01
@desc:
"""
import keras
import tqdm
import numpy as np
import os
import gdal
import cv2
from segmentation_models.losses import cce_jaccard_loss
from segmentation_models.metrics import iou_score, f1_score

def read_p(path, model):

    dataset = gdal.Open(path)

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    print("files:{},shape:{}/{}/{}".format(path, rows, cols, bands))

    res = np.zeros(shape=(rows, cols), dtype=np.uint8)

    band_r = dataset.GetRasterBand(1)
    band_g = dataset.GetRasterBand(2)
    band_b = dataset.GetRasterBand(3)



    size = 1152

    for i in tqdm.tqdm(range(0, rows - size, size // 2)):
        for j in range(0, cols - size, size // 2):

            r = band_r.ReadAsArray(j, i, size, size)
            if np.sum(r == 0) / r.size > 0.5:  # 统计黑色像素比例，大于0.5不加入训练集
                continue
            g = band_g.ReadAsArray(j , i, size, size)
            b = band_b.ReadAsArray(j, i , size, size)

            img = cv2.merge([r,g,b]) # 注意为RGB，需要归一化
            img = cv2.resize(img, (384, 384))  # 缩放
            img = img / 255.0



            imgs = np.array([img])
            pred = model.predict(imgs)
            pred = np.argmax(pred[0],axis=-1)
            pred = cv2.resize(pred, (size, size), interpolation=cv2.INTER_NEAREST)

            res[i + size // 4:i + size // 4 * 3, j + size // 4:j + size // 4 * 3] = pred[size // 4: size // 4 * 3, size // 4: size // 4 * 3]

    cv2.imwrite(
        "output/res/{}_predict.png".format(os.path.splitext(path)[0].split("/")[-1]), res)

if __name__ == '__main__':

    model = keras.models.load_model("output/tc_arg_model.h5",custom_objects={"iou_score":iou_score,'cce_jaccard_loss': cce_jaccard_loss})
    model.summary()

    read_p("../data/jingwei_round1_test_a_20190619/image_3.png", model)
    read_p("../data/jingwei_round1_test_a_20190619/image_4.png", model)
