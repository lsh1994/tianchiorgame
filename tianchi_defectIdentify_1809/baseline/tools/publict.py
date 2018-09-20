"""
@file: publict.py
@author: lishihang 
@software: PyCharm
@time: 2018/09/18
"""
import os
from pyecharts import Line

def show(path):
    d=os.listdir(path)
    d_len=[len(os.listdir(os.path.join(path,i))) for i in d]

    line = Line(path)
    line.add(path, d, d_len, mark_point=["average", "max", "min"],xaxis_rotate=40)
    return line

line=show(r"D:\TianChi\defectIdentify\sample\train\其他\\")
line.render(path="temp.html")