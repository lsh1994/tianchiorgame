"""
@author: LiShiHang
@software: PyCharm
@file: dataflow.py
@time: 2019/2/20 9:45
@desc:
"""
import numpy as np
import pandas as pd
import gdal
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class TrainFlow:

    def __init__(self, label, tif, size, bands):
        self.label_data = pd.read_csv(label, header=None).values

        print(Counter(self.label_data[:,0]))

        self.label_length = self.label_data.shape[0]
        self.dataset = gdal.Open(tif)
        self.size = size
        self.label_dict = {'玉米': 0, '大豆': 1, '水稻': 2, "其他": 3}
        self.bands = bands

    def _get_cell(self, pos_x, pos_y):

        try:
            output = []
            for i in self.bands:
                band = self.dataset.GetRasterBand(i)
                t = band.ReadAsArray(int(pos_x - self.size / 2),
                                     int(pos_y - self.size / 2), self.size, self.size)
                output.append(t)
            img = np.moveaxis(np.array(output), 0, 2)
        except BaseException:
            return None
        return img

    def _get_cells(self, pos_start, pos_end):
        imgs = []
        labels = []
        for i in range(pos_start, pos_end):
            temp = self.label_data[i, :]
            img = self._get_cell(temp[1], temp[2])
            if img is None:
                continue
            imgs.append(img)
            label = np.zeros(4)
            label[self.label_dict[temp[0]]] = 1
            labels.append(label)
        imgs = np.array(imgs)
        labels = np.array(labels)
        return imgs, labels

    def get_batch(self, batch_size, enhance=False):

        fea_len = self.label_data.shape[0]

        seq = iaa.Sequential([
            # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Crop(px=(0, 6)),
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            iaa.Flipud(0.5),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True)

        while True:
            np.random.shuffle(self.label_data)
            for pos_start in range(0, fea_len, batch_size):
                pos_end = min(pos_start + batch_size, fea_len)
                imgs, labels = self._get_cells(pos_start, pos_end)
                if enhance:
                    imgs = seq.augment_images(imgs)
                yield imgs, labels


if __name__ == '__main__':
    obj = TrainFlow(label="data/train_enhance.txt",
                    tif=r"E:\机器学习竞赛\baidu_dianshi\rgb_data.tif",
                    size=25, bands=[1, 2, 3])

    t = obj.get_batch(24, True).__next__()
    print(t[0].shape, t[1].shape)
    fig, ax = plt.subplots(
        4, 6, sharex=True, sharey=True, figsize=(
            10, 8), num="样本实例")
    for i in range(4):
        for j in range(6):
            img = t[0][i * 4 + j, :, :, :]
            label = list(obj.label_dict.keys())[np.argmax(t[1][i * 4 + j])]
            ax[i][j].set_title(label)
            ax[i][j].imshow(img)
            ax[i][j].axis('off')
    fig.tight_layout()
    plt.show()
