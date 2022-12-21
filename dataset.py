import torch
import random
import numpy as np
import os
import glob
from utils import *

class DTTDMECDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode='Train'):
        super().__init__()
        self.dataset_path = cfg.dataset_path
        self.add_gaussian = cfg.add_gaussian
        self.mode = mode
        # load in all precomputed features
        self.img_features, self.points_features, self.labels = self._load_precomputed_features()
    
    def _load_precomputed_features(self):
        if self.mode == 'Train' or self.mode == 'Test':
            data = np.load(os.path.join(self.dataset_path, self.mode + ".npy"), allow_pickle = True)
            img = data.item()["img"]
            points = data.item()["points"]
            label = np.expand_dims(data.item()["class"], axis=1)
        elif self.mode == 'All':
            train_data = np.load(os.path.join(self.dataset_path, "Train.npy"), allow_pickle = True)
            if self.add_gaussian:
                batch, row, col, ch = train_data.item()["img"].shape
                mean = 0.0
                sigma = 1.0
                gauss = np.random.normal(mean, sigma,(batch, row, col, ch))
                gauss = gauss.reshape(batch, row, col, ch)
                train_data.item()["img"] = train_data.item()["img"] + gauss
            test_data = np.load(os.path.join(self.dataset_path, "Test.npy"), allow_pickle = True)
            img = np.concatenate([train_data.item()["img"], test_data.item()["img"]], axis=0)
            points = np.concatenate([train_data.item()["points"], test_data.item()["points"]], axis=0)
            label = np.expand_dims(np.concatenate([train_data.item()["class"], test_data.item()["class"]], axis=0), axis=1)
        return img, points, label
    
    def __getitem__(self, ind):
        img = self.img_features[ind]
        pc = self.points_features[ind]
        label = self.labels[ind]
        
        result = {
            IMAGE_KEY: torch.FloatTensor(img),
            POINT_KEY: torch.FloatTensor(pc),
            LABEL_KEY: torch.tensor(label)
        }
        return result

    def __len__(self):
        return len(self.labels)
    
def split_dataset(dataset, rate=0.95):
    train_n = int(rate * len(dataset))
    test_n = len(dataset) - train_n
    train_set, val_set = torch.utils.data.random_split(dataset, [train_n, test_n])
    return train_set, val_set