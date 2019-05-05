import numpy as np
from osgeo import gdal
import keras
from multiprocessing import Pool
import multiprocessing
import os
import datetime
import cv2

np.set_printoptions(threshold=np.inf)  # 使print大量数据不用符号...代替而显示所有


def get_cell(pos_x, pos_y, size):

    up = int(pos_x - size / 2)
    left = int(pos_y - size / 2)

    up = int(np.
             clip(up, 0, dataset.RasterXSize - size))
    left = int(np.clip(left, 0, dataset.RasterYSize - size))

    try:
        output = []
        for i in bands:
            band = dataset.GetRasterBand(i)
            t = band.ReadAsArray(up, left, size, size)
            output.append(t)
        img = np.moveaxis(np.array(output), 0, 2)
    except BaseException:
        return None
    return img


size = 25
labels_key = [20, 40, 60, 0]
bands = [1, 2, 3]

model = keras.models.load_model("output/model_save.h5")
dataset = gdal.Open(
    r"E:\机器学习竞赛\baidu_dianshi\rgb_data.tif")


def render(num):

    res = np.zeros(shape=(10073, 8905), dtype=np.uint8)

    col = 5

    left = (num % col) * 10073
    top = (num // col) * 8905

    print(num, top, left)

    # step = 7
    # for i in range(0, 10073 + step, step):
    #     for j in range(0, 8905 + step, step):
    #         img = get_cell(i + left, j + top, size)
    #         if img is None:
    #             continue
    #         imgs = np.array([img])
    #         result = model.predict(imgs)
    #         x1 = max(i - 3, 0)
    #         x2 = min(i + 4, 10073)
    #         y1 = max(j - 3, 0)
    #         y2 = min(j + 4, 8905)
    #         res[x1:x2, y1:y2] = labels_key[np.argmax(result, 1)[0]]
    #     print("processing pid:{} {}/{}".format(os.getpid(), i, 10073))

    step = 3
    for i in range(0, 10073 + step, step):
        for j in range(0, 8905 + step, step):
            img = get_cell(i + left, j + top, size)
            if img is None:
                continue
            imgs = np.array([img])
            result = model.predict(imgs)
            x1 = max(i - 1, 0)
            x2 = min(i + 2, 10073)
            y1 = max(j - 1, 0)
            y2 = min(j + 2, 8905)
            res[x1:x2, y1:y2] = labels_key[np.argmax(result, 1)[0]]
        print("processing pid:{} {}/{}".format(os.getpid(), i, 10073))

    cv2.imwrite("imgres/test_result_{}.tif".format(num), res.T)


if __name__ == '__main__':

    start = datetime.datetime.now()

    pool = Pool(multiprocessing.cpu_count())
    print(multiprocessing.cpu_count())
    img_list = range(10)
    pool.map(render, img_list)

    print(datetime.datetime.now() - start)
