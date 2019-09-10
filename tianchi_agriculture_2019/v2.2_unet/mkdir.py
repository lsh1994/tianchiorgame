"""
@author: LiShiHang
@file: mkdir.py
@software: PyCharm
@time: 2019/7/16 23:21
@desc: 
"""
import shutil,os

def min_mkdir(s):

    if os.path.exists(s):
        shutil.rmtree(s)
    os.mkdir(s)

if __name__ == '__main__':
    min_mkdir("data/")
    min_mkdir("data/train/")
    min_mkdir("data/train/imgs")
    min_mkdir("data/train/labels")
    min_mkdir("output/")
    min_mkdir("output/res/")