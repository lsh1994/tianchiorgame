import numpy as np
import cv2
import imutils
from skimage.segmentation import mark_boundaries
import random
import glob
import matplotlib.pyplot as plt

def data_augment(im, seg):

    def rotate(im, seg):

        angle = np.random.randint(0, 20)
        scale = np.random.randint(8, 12) / 10

        img_w = im.shape[1]
        img_h = im.shape[0]

        M_rotate = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), angle, scale)
        im = cv2.warpAffine(im, M_rotate, (img_w, img_h))
        seg = cv2.warpAffine(seg, M_rotate, (img_w, img_h), flags=cv2.INTER_NEAREST)  # 注意插值方式
        return im, seg

    def translation(im, seg, rate):
        x_offset = np.random.randint(-im.shape[0] * rate, im.shape[0] * rate)
        y_offset = np.random.randint(-im.shape[1] * rate, im.shape[1] * rate)

        M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])

        im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))
        seg = cv2.warpAffine(seg, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_NEAREST)  # 注意插值方式)
        return im, seg

    im = im.copy()
    seg = seg.copy()

    if np.random.random() < 0.25:
        im, seg = translation(im, seg,0.2)

    if np.random.random() < 0.25:
        im, seg = rotate(im, seg)

    if np.random.random() < 0.25:
        im = cv2.flip(im, 1)
        seg = cv2.flip(seg, 1)

    if np.random.random() < 0.25:
        im = cv2.flip(im, 0)
        seg = cv2.flip(seg, 0)

    return im, seg

def label2color(label,colors):

    im = np.zeros(shape=(label.shape[0], label.shape[1], 3))
    for i in range(classes):
        im[label == i] = colors[i]
    return im.astype(np.uint8)


def get_flow(batch, size=1152, argument=True, shuffle=True):

    def one_hot(seg,classes=4):
        oh = np.zeros(shape=(seg.shape[0],seg.shape[1],classes),dtype=np.uint8)
        for i in range(classes):
            oh[:,:,i]=(seg==i).astype(int)
        return oh

    def get_imgs_labels(src_batch):

        X = []
        Y = []

        for p1 in src_batch:

            p2 = p1.replace("imgs", "labels")

            im = cv2.imread(p1)
            seg = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)

            # 其他数据增强
            if argument:
                im, seg = data_augment(im, seg)
            seg = one_hot(seg)

            # 随机裁剪，变成 size*size*8,size*size*1
            xx = np.random.randint(0, im.shape[0] - size + 1)
            yy = np.random.randint(0, im.shape[1] - size + 1)
            im = im[xx:xx + size, yy:yy + size]
            seg = seg[xx:xx + size, yy:yy + size]

            im = cv2.resize(im, (384, 384))  # 缩放
            seg = cv2.resize(seg, (384, 384), interpolation=cv2.INTER_NEAREST)

            X.append(im)
            Y.append(seg)
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    src = list(glob.glob("data/train/imgs/*.png"))

    while True:
        if shuffle:
            random.shuffle(src)
        for i in range(0, len(src), batch):
            src_batch = src[i:i + batch]  # 文件索引

            X, Y = get_imgs_labels(src_batch)

            X = X[:,:,:,[2,1,0]] # 注意将BGR转为RGB
            X = X/255.0 # 归一化

            yield X, Y

if __name__ == '__main__':

    classes = 4

    colors = np.random.randint(0, 256, (classes, 3))

    G = get_flow(32, 1152, True)

    while True:
        a, b = G.__next__()
        print(a.shape,b.shape)

        # fig, ax = plt.subplots(4, 8,  num="标签")
        # fig2, ax2 = plt.subplots(4, 8, num="影像")
        #
        # for i in range(4):
        #     for j in range(8):
        #         ax[i][j].imshow(label2color(b[i * 8 + j,:,:,0],colors))
        #         ax[i][j].set_xticks([])
        #         ax[i][j].set_yticks([])
        #
        #         ax2[i][j].imshow(a[i * 8 + j])
        #         ax2[i][j].set_xticks([])
        #         ax2[i][j].set_yticks([])
        #
        # fig.tight_layout()
        # plt.show()



