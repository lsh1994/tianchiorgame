import pandas as pd
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from keras.models import load_model
import cv2

np.set_printoptions(threshold=np.inf)  # 使print大量数据不用符号...代替而显示所有


def get_cell(pos_x, pos_y):

    try:
        output = []
        for i in bands:
            band = dataset.GetRasterBand(i)
            t = band.ReadAsArray(int(pos_x - size / 2),
                                 int(pos_y - size / 2), size, size)
            output.append(t)
        img = np.moveaxis(np.array(output), 0, 2)
    except BaseException:
        return None
    return img


dataset = gdal.Open(
    r"D:\baidu_dianshi\验证集原始图像_8波段.tif")
bands = [i + 1 for i in range(8)]
size = 5
labels_key = [20, 60, 40]


def render(name):
    # todo: 使用多线程、向量运算加速
    model = load_model("data/model_save.h5")

    res = np.zeros(
        shape=(
            dataset.RasterXSize,
            dataset.RasterYSize),
        dtype=np.uint8)

    step = 5
    t = size // 2 + 1
    for i in tqdm(range(t, dataset.RasterXSize, step)):

        for j in range(t, dataset.RasterYSize, step):
            # print(i,j)
            img = get_cell(i, j)
            if img is None:
                continue
            imgs = np.array([img])
            result = model.predict(imgs)[0]
            if np.max(result) < 0.5:  # 拒判
                s = 0
            s = labels_key[np.argmax(result)]
            res[i - t:i + t, j - t:j + t] = s

    cv2.imwrite("{}_t.tif".format(name), res.T)


if __name__ == '__main__':
    render("cnn4321")
