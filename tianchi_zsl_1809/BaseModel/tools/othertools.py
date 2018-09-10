"""
@file: othertools.py
@author: lishihang 
@software: PyCharm
@time: 2018/09/08
"""
import pandas as pd


def eval():
    truelabels = pd.read_csv("../data/zsl_validate.csv", sep='\t', header=None)
    truelabels=truelabels.iloc[:,1].values
    predlabels = pd.read_csv("../output/submittest.txt", sep='\t', header=None)
    predlabels = predlabels.iloc[:, 1].values
    from sklearn.metrics import accuracy_score
    print(accuracy_score(truelabels,predlabels))



if __name__ == '__main__':

    eval()