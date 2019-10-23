import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import cv2
import imutils
import os
import config

p = pd.read_csv("output/submit_best.csv")

t = pd.read_csv(
    "backup/efficientb3_0.84828717/train.csv")[["FileName", "type"]]

# print(p.head())
# print(t.head())

predict_label = p.type
true_label = t.type
print(classification_report(true_label, predict_label))

C2 = confusion_matrix(true_label, predict_label)
print(C2)
print(f"f1={f1_score(true_label, predict_label,average='macro')},acc={accuracy_score(true_label, predict_label)}")
print()

for i, k in p[predict_label != true_label].iterrows():
    # print(i,k.FileName,k.type)
    img = cv2.imread(os.path.join(config.base, "Train", k.FileName))
    print(
        f"文件：{p.FileName[i]},预测：{p.type[i],config.name_dict[p.type[i]]},真实标记：{t.type[i],config.name_dict[t.type[i]]}")
    cv2.imshow("wrong", imutils.resize(img, height=600))
    cv2.waitKey(0)
