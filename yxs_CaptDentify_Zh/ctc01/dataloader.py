from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import random
from Utilss import *
from torchvision.transforms.functional import to_tensor

import config


class CaptDataset(Dataset):
    def __init__(self,csv,transforms,dir,is_test=False):
        self.data = pd.read_csv(csv)
        self.transforms = transforms
        self.is_test = is_test
        self.dir = dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        filename,label = self.data.iloc[item,:]

        image = cv2.cvtColor(cv2.imread(self.dir+filename),cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        image = to_tensor(image)

        if self.is_test:
            return image,filename
        else:
            if len(label) < config.maxL:
                label = list(label)
                while len(label) < config.maxL: label.insert(random.randint(0,len(label)+1),"-")
                label = "".join(label)
            # while len(label) < config.maxL: label = "-" + label

            target = torch.tensor([config.key_words.find(x) for x in label], dtype=torch.long)
            input_length = torch.full(size=(1,), fill_value=config.n_input_length, dtype=torch.long)
            target_length = torch.full(size=(1,), fill_value=config.maxL, dtype=torch.long)
            return image, target, input_length, target_length
