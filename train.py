import itertools
import os
import time
import argparse
import json
#import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MLP
from dataset import DTTDMECDataset
from utils import *
from tqdm import tqdm

def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=2022)
    parser.add_argument('--training_epoch', default=20)
    parser.add_argument('--dataset_path', default='dataset')
    parser.add_argument('--config', default='config.json')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(a.seed)
        torch.cuda.empty_cache()
    else:
        pass

    training_dataset = DTTDMECDataset(cfg = a, mode = "Train")

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=h.batchsize, shuffle=True, num_workers=h.num_workers)
    
    mlp = MLP(
        in_sizes=h.input_sizes,
        out_sizes=h.output_sizes,
        activation=h.activation,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    
    for epoch in range(0, a.training_epoch):
        print(f'Starting epoch {epoch}')
        current_loss = 0.0
        for i, data in tqdm(enumerate(training_dataloader)):
            imgs, points, labels = data[IMAGE_KEY], data[POINT_KEY], data[LABEL_KEY]
            optimizer.zero_grad()
            flattened_imgs = torch.flatten(imgs, 1)
            flattened_points = torch.flatten(points, 1)
            inputs = torch.cat((flattened_imgs, flattened_points), dim=1)
            outputs = mlp(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 500))
                current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')




if __name__ == '__main__':
    main()