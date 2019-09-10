from osgeo import gdal
import numpy as np
import cv2
import os
import imutils
import glob
import pandas as pd
import datetime
import tqdm
import shutil


def split_train(fp,split_size, fp_label=None):
    if not fp_label:
        fp_label = str(fp).replace(".png", "_label.png")

    dataset = gdal.Open(fp)
    dataset_label = gdal.Open(fp_label)

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount
    print("files:{},shape:{}/{}/{}".format(fp, rows, cols, bands))

    band_r = dataset.GetRasterBand(1)
    band_g = dataset.GetRasterBand(2)
    band_b = dataset.GetRasterBand(3)
    labels = dataset_label.GetRasterBand(1)

    for i in tqdm.tqdm(range(0, rows - split_size, split_size)):

        for j in range(0, cols - split_size, split_size):

            r = band_r.ReadAsArray(j, i, split_size, split_size)
            if np.sum(r == 0) / r.size > 0.5:  # 统计黑色像素比例，大于0.5不加入训练集
                continue
            g = band_g.ReadAsArray(j, i, split_size, split_size)
            b = band_b.ReadAsArray(j, i, split_size, split_size)
            label = labels.ReadAsArray(j, i, split_size, split_size)
            img = cv2.merge([b, g, r])

            fn = os.path.basename(fp).rsplit(".")[0]
            file_save_path = os.path.join(
                "data/train/imgs/", "{}_{}_{}.png".format(fn, i, j))
            file_label_save_path = os.path.join(
                "data/train/labels/", "{}_{}_{}.png".format(fn, i, j))
            cv2.imwrite(file_save_path, img)
            cv2.imwrite(file_label_save_path, label)


if __name__ == '__main__':

    start = datetime.datetime.now()
    split_train(
        fp="../data/jingwei_round1_train_20190619/image_1.png",
        split_size=1200)
    split_train(
        fp="../data/jingwei_round1_train_20190619/image_2.png",
        split_size=1200)
    end = datetime.datetime.now()
    print("start:{},end:{},last:{}".format(start, end, end - start))
