import torch
import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

class IMetDataset(Dataset):

    def __init__(self,
                 df,
                 images_dir,
                 n_classes,
                 transforms, is_test=False
                 ):
        self.df = df
        self.images_dir = images_dir
        self.n_classes = n_classes
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]

        img_path = os.path.join(self.images_dir, cur_idx_row["FileName"])

        img = None
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            pass
        if img is None:
            img = np.zeros(shape=(224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img)
        img = self.transforms(img)

        if self.is_test:
            return img, cur_idx_row["FileName"]
        else:
            return img, int(cur_idx_row["type"]) - 1