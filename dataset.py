import torch
import random
import numpy as np
import os
import glob
from utils import *

class DTTDMECDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode='train'):
        super().__init__()
        self.dataset_path = cfg.dataset_path
        self.mode = mode
        # load in all precomputed features
        self.img_features, self.points_features, self.labels = self._load_precomputed_features()
    
    def _load_precomputed_features(self):
        data = np.load(os.path.join(self.dataset_path, self.mode + ".npy"), allow_pickle = True)
        img = data.item()["img"]
        points = data.item()["points"]
        label = np.expand_dims(data.item()["class"], axis=1)
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