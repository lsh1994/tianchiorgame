"""
@author: LiShiHang
@software: PyCharm
@file: fspost_mp.py
@time: 2019/3/3 16:08
@desc:
"""
import numpy as np
from osgeo import gdal
from multiprocessing import Pool
import multiprocessing
import os
import datetime
import cv2
from scipy import stats

np.set_printoptions(threshold=np.inf)  # 使print大量数据不用符号...代替而显示所有


def get_cell(pos_x, pos_y, size):
    up = int(pos_x - size / 2)
    left = int(pos_y - size / 2)

    up = int(np.
             clip(up, 0, dataset.RasterXSize - size))
    left = int(np.clip(left, 0, dataset.RasterYSize - size))

    try:
        img = dataset.ReadAsArray(up, left, size, size)
    except BaseException:
        return None
    return img


size = 155
dataset = gdal.Open(
    r"imgres/test_result_combine.tif")


def render(num):
    res = np.zeros(shape=(10073, 8905), dtype=np.uint8)

    col = 5

    left = (num % col) * 10073
    top = (num // col) * 8905

    step = 3
    for i in range(0, 10073+step, step):
        for j in range(0, 8905+step, step):
            img = get_cell(i + left, j + top, size)
            if img is None:
                continue
            imgs = np.array(img)

            zs = stats.mode(imgs.reshape(-1))[0][0]

            x1 = max(i - 1, 0)
            x2 = min(i + 2, 10073)
            y1 = max(j - 1, 0)
            y2 = min(j + 2, 8905)
            res[x1:x2, y1:y2] = zs

        print("processing pid:{} {}/{}".format(os.getpid(), i, 10073))

    cv2.imwrite("imgres/fspost_result_{}.tif".format(num), res.T)


if __name__ == '__main__':
    start = datetime.datetime.now()

    pool = Pool(multiprocessing.cpu_count())
    print(multiprocessing.cpu_count())
    img_list = range(10)
    pool.map(render, img_list)

    print(datetime.datetime.now() - start)

    # render(0)
