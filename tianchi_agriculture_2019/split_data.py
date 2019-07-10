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


def split_train(file_base, file, split_size):
    """
    后期加入多线程
    """
    file_path = [file_base + i for i in file]

    for fp in file_path:
        dataset = gdal.Open(fp + ".png")
        dataset_label = gdal.Open(fp + "_label.png")

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

                file_save_path = os.path.join(
                    "data/train/imgs/", "{}_{}_{}.jpg".format(fp.split("/")[-1], i, j))
                file_label_save_path = os.path.join(
                    "data/train/labels/", "{}_{}_{}.jpg".format(fp.split("/")[-1], i, j))
                cv2.imwrite(file_save_path, img)
                cv2.imwrite(file_label_save_path, label)


def label_statistics(train_labels, classes):

    def get_label_weight(path):
        img_view = cv2.imread(path, cv2.IMREAD_GRAYSCALE).flatten()
        labels_count = np.zeros(shape=(classes,))
        for i in range(classes):
            labels_count[i] = np.sum(img_view == i)
        labels_count = np.around(labels_count / len(img_view), decimals=2)
        return labels_count

    res = []
    for i in tqdm.tqdm(glob.glob(os.path.join(train_labels, "*.jpg"))):
        res.append([os.path.abspath(i).replace("labels", "imgs"), ] +
                   get_label_weight(i).tolist())
    res = pd.DataFrame(res)
    res.to_csv("data/train/labels2txt.txt", index=None, header=None, sep="\t")
    # print(res.head())

def min_mkdir(s):

    if os.path.exists(s):
        shutil.rmtree(s)
    os.mkdir(s)

if __name__ == '__main__':

    min_mkdir("data/train")
    min_mkdir("data/train/imgs")
    min_mkdir("data/train/labels")

    start = datetime.datetime.now()
    split_train(
        file_base="data/jingwei_round1_train_20190619/",
        file=[
            "image_1",
            "image_2"],
        split_size=224)
    end = datetime.datetime.now()
    print("start:{},end:{},last:{}".format(start, end, end - start))
    # start:2019-07-06 09:27:33.135577,end:2019-07-06
    # 09:43:43.219755,last:0:16:10.084178

    label_statistics("data/train/labels", classes=4)
