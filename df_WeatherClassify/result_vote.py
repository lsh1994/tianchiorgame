import os
import glob
import pandas as pd
import numpy as np
from collections import Counter

pd.set_option('display.width',None)
np.set_printoptions(threshold=np.inf)

fp_list = ["backup/balance/densenet121_0.84602296_epoch250/output__all_last.csv",
           "backup/balance/efficientb3_0.85738826_epoch150/output__all_last.csv",
           "backup/balance/efficientb3_0.86144787_epoch250/output__all_last.csv",
           "backup/balance/xception_0.85121089_epoch250/output__all_last.csv"]

#####################################
data = pd.read_csv(fp_list[0])
names = data.iloc[:,0].tolist()
values = data.iloc[:,1:].values

for fp in fp_list[1:]:
    d = pd.read_csv(fp).iloc[:,1:].values
    values = np.add(values,d)


###################################
new_res = pd.DataFrame({
    "FileName": names,
    "type": np.argmax(values,1)
})
new_res.type = new_res.type.apply(lambda x: x + 1)
new_res.to_csv("output/submit_vote.csv", index=None)
print(new_res.head())

print(new_res.type.value_counts())
