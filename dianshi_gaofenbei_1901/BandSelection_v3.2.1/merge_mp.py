"""
@author: LiShiHang
@software: PyCharm
@file: merge_mp.py
@time: 2019/2/1 14:55
@desc:
"""
import numpy as np
import cv2

row = 2
col = 5

res_index = np.zeros(
    shape=(
        17810,
        50365), dtype=np.uint8)

for i in range(row):
    for j in range(col):
        src = r"imgres\test_result_{}.tif".format(i * 5 + j)
       
        print(src)

        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)

        res_index[i * 8905:(i + 1) * 8905,
                  j * 10073:(j + 1) * 10073] = img

res_index = res_index[:, 0:50362]
print(res_index.shape)

cv2.imwrite("imgres/test_result_combine.tif", res_index)
