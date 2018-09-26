"""
@file: views.py
@author: lishihang 
@software: PyCharm
@time: 2018/09/19
"""
from pyecharts import Line,Scatter
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


def drawlog(dic):
    """

    :param his: keras fit返回的的history对象字典
    :return:
    """
    line = Scatter("log")
    for label, value in dic.items():
        line.add(label, [i for i in range(len(value))], value,symbol_size=3)
    line.render(path="output/temp.html")

def wucha(fl,tl,label):
    """
    生成结果
    :param fl: 预测的值
    :param tl: 真实值
    :param label: 标签
    :return:
    """
    print(classification_report(tl, fl))
    print("acc:",accuracy_score(tl,fl))
    mat = confusion_matrix(tl, fl)
    sns.heatmap(mat,annot=True, square=True, fmt="d", xticklabels=label, yticklabels=label)
    plt.show()