import os
import glob
import pandas as pd
import numpy as np
from collections import Counter

pd.set_option('display.width',None)
np.set_printoptions(threshold=np.inf)

fp_list = ["backup/densenet121_0.84602296_epoch250/submit_last.csv",
           "backup/efficientb3_0.85738826_epoch150/submit_last.csv",
           "backup/efficientb3_0.86144787_epoch250/submit_last.csv",
           "backup/xception_0.85121089_epoch250/submit_last.csv"]

fp_list += glob.glob("backup/others/*.csv")
print(fp_list,len(fp_list))
#####################################
data = pd.read_csv(fp_list[0])
names = data.iloc[:,0].tolist()
values = data.iloc[:,1:].values

for fp in fp_list[1:]:
    d = pd.read_csv(fp).iloc[:,1:].values
    values = np.concatenate((values,d),axis=1)


###################################
new_res = pd.DataFrame({
    "FileName": names,
    "type": [Counter(values[i,:]).most_common(1)[0][0] for i in range(values.shape[0])]
})
new_res.to_csv("output/submit_vote_by_class.csv", index=None)
print(new_res.head())

print(new_res.type.value_counts())
