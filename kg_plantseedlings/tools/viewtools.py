"""
@file: viewtools.py
@author: lishihang 
@software: PyCharm
@time: 2018/09/21
"""

import os

from pyecharts import Line

def show(path):
    d=os.listdir(path)
    d_len=[len(os.listdir(os.path.join(path,i))) for i in d]

    line = Line()
    line.add("train", d, d_len, mark_point=["average", "max", "min"],xaxis_rotate=40, )
    line.render(path="temp.html")



if __name__ == '__main__':
    show(r"D:\TianChi\kgPlantSeedlings\train\\")