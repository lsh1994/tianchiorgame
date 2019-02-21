"""
@author: LiShiHang
@software: PyCharm
@file: clear_data.py
@time: 2019/1/28 11:10
@desc:
"""
import pandas as pd
import numpy as np
from osgeo import gdal
from tqdm import tqdm

np.set_printoptions(threshold=np.inf)  # 使print大量数据不用符号...代替而显示所有


def data_streamlined():

    s = pd.read_csv("data/sample_train.txt")
    s = s.values

    print(s[:3, :])  # 样例输出
    print(s.shape)  # 输出形状

    # print(s[s[:, 3] != 3])  # 尺寸全等于3

    res = s[:, [2, 5, 6]]
    res[:, -1] = [round(-i) for i in res[:, -1]]
    res[:, -2] = [round(i) for i in res[:, -2]]

    res = pd.DataFrame(res, index=None, columns=None)
    res.to_csv("data/train.txt", header=None, index=None)

    return res


def get_tif_data(res):
    tif_data = []
    for _, row in tqdm(list(res.iterrows())):
        img = get_cell(row[1], row[2])
        tif_data.append([img, labels_key[row[0]]])
    tif_data = np.array(tif_data)
    return tif_data


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
    r"D:\baidu_dianshi\GF6_WFV_E127.9_N46.8_20180823_L1A1119838015.tif")
bands = [i + 1 for i in range(8)]
size = 5
labels_key = {'玉米': 0, '水稻': 1, '大豆': 2}

if __name__ == '__main__':

    res = data_streamlined()

    tif_data = get_tif_data(res)
    print(tif_data.shape)
    np.save("data/train_raw_{}x{}.npy".format(size, size), tif_data)
    print(tif_data[0])
    print("finish.")
