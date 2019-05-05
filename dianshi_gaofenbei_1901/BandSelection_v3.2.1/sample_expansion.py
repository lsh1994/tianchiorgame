"""
@author: LiShiHang
@software: PyCharm
@file: sample_expansion.py
@time: 2019/2/20 9:06
@desc:
"""
import pandas as pd
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from collections import Counter
from pyod.models.iforest import IForest


np.set_printoptions(threshold=np.inf)  # 使print大量数据不用符号...代替而显示所有


dataset = gdal.Open(
    r"E:\机器学习竞赛\baidu_dianshi\rgb_data.tif")
labels_key = {'玉米': 0, '大豆': 1, '水稻': 2, '其他': 3}


def data_streamlined():

    s = pd.read_csv("data/sample_train.txt")
    s = s.values

    print(s[:3, :])  # 样例输出
    print(s.shape)  # 输出形状

    print(s[s[:, 3] != 3])  # 尺寸全等于3

    res = s[:, [2, 5, 6]]
    res[:, -1] = [round(-i) for i in res[:, -1]]
    res[:, -2] = [round(i) for i in res[:, -2]]

    res = pd.DataFrame(res, index=None, columns=None)
    res.to_csv("data/train.txt", header=None, index=None)


def get_cell(pos_x, pos_y, size):

    try:
        output = []
        for i in [1,2,3]:
            band = dataset.GetRasterBand(i)
            t = band.ReadAsArray(int(pos_x - size / 2),
                                 int(pos_y - size / 2), size, size)
            output.append(t)
        img = np.moveaxis(np.array(output), 0, 2)
    except BaseException:
        return None
    return img


def add_other_class(num, size,pad):
    res = pd.read_csv("data/train.txt", header=None).values
    tif_data = []
    for r in tqdm(range(res.shape[0])):
        img = get_cell(res[r][1], res[r][2], size)
        if img is None:
            print("img NOT Exist.", res[r])
            continue
        img = img.reshape(-1).tolist()
        tif_data.append([labels_key[res[r][0]]] + img)
    tif_data = np.array(tif_data)
    print(tif_data.shape)

    np.random.shuffle(tif_data)
    clf = IForest()
    clf.fit(tif_data[:, 1:])

    i = 0
    pos = []
    false_num = 0
    while True:
        ix = np.random.randint(
            pad, dataset.RasterXSize-pad)
        iy = np.random.randint(pad, dataset.RasterYSize-pad)
        t = get_cell(ix, iy, size)
        if t is None:
            continue
        t = t.reshape(1, -1)
        y_test_pred = clf.predict(t)[0]  # outlier labels (0 or 1)
        if y_test_pred == 1:
            i += 1
            pos.append(["其他"] + [ix, iy])
            print("{}/{} added.".format(i, num))
        else:
            false_num += 1
            print("{}/{} is not include {}.{}. false_num: {}".format(i,
                                                                     num, ix, iy, false_num))

        if i == num:
            break
    pos = np.concatenate(
        (res, np.array(pos)), axis=0)
    print(Counter(pos[:, 0]))

    pd.DataFrame(pos).to_csv(
        "data/train_enhance.txt", index=None, header=None)

    pos[:, 2] = -1 * (pos[:, 2].astype(np.int))
    pd.DataFrame(pos).to_csv(
        "data/train_enhance_view.txt", index=None, header=None)


if __name__ == '__main__':

    data_streamlined()
    add_other_class(num=3000, size=7, pad=64)
    print("finish.")
