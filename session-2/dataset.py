import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):
        super().__init__()
        self.df = pd.read_csv(labels_path)
        self.images_path = images_path
        self.len = len(list(os.listdir(images_path)))
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        suite_id,sample_id,code,_,_ = self.df.loc[idx, :]
        label = code
        image = Image.open(self.images_path + f"input_{suite_id}_{sample_id}_{code}.jpg")
        
        if self.transform:
            image = self.transform(image)
        
        return image, code-1
