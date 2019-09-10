"""
@author: LiShiHang
@file: predict.py
@software: PyCharm
@time: 2019/7/7 11:15
@desc:
"""
import keras
import glob
import numpy as np
import os
import gdal
import cv2
import tqdm


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

    size = 224
    step = 33
    for i in tqdm.tqdm(range(112, rows - size, step)):

        for j in range(112, cols - size, step):

            # i,j为中心点

            r = band_r.ReadAsArray(j-112, i-112, size, size)
            if np.sum(r == 0) / r.size > 0.5:  # 统计黑色像素比例，大于0.5不加入训练集
                continue
            g = band_g.ReadAsArray(j-112, i-112, size, size)
            b = band_b.ReadAsArray(j-112, i-112, size, size)

            img = cv2.merge([b, g, r])

            img = cv2.resize(img, (size, size)) / 255.0

            imgs = np.array([img])
            pred = model.predict(imgs)

            res[i - step//2:i + step//2+1, j - step//2:j + step//2+1] = np.argmax(pred, axis=1)[0]

    cv2.imwrite(
        "output/pred/{}_predict.png".format(os.path.splitext(path)[0].split("/")[-1]), res)


if __name__ == '__main__':


    model = keras.models.load_model("output/0709_2_model.h5")
    model.summary()

    read_p("../data/jingwei_round1_test_a_20190619/image_3.png", model)
    read_p("../data/jingwei_round1_test_a_20190619/image_4.png", model)
