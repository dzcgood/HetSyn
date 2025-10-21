from torch.utils.data import Dataset
import torch
import numpy as np


class SHD(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):   
        x = torch.from_numpy(np.load(self.data_paths[index]))
        y_ = self.data_paths[index].split('_')[-1]
        y_ = int(y_.split('.')[0])
        y = torch.tensor([y_])
        return x, y
    