import itertools
import os
import time
import argparse
import json
#import torchaudio
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model import MLP
from dataset import DTTDMECDataset, split_dataset
from utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def main():
    print('Initializing Training Process..')
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--training_epoch', default=100, type=int)
    parser.add_argument('--dataset_path', default='dataset', type=str)
    parser.add_argument('--config', default='config.json', type=str)
    parser.add_argument('--checkpoint_path', default='ckpt', type=str)
    parser.add_argument('--summary_interval', default=50, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(a.seed)
        torch.cuda.empty_cache()
    else:
        pass

    # device = torch.device('cuda:{:d}'.format(0))
    device = torch.device('cpu')

    # training_dataset = DTTDMECDataset(cfg = a, mode = "Train")
    # testing_dataset = DTTDMECDataset(cfg = a, mode = "Test")
    training_dataset, testing_dataset = split_dataset(DTTDMECDataset(cfg = a, mode = "All"), 0.96)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=h.batchsize, shuffle=True, num_workers=h.num_workers)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=False, num_workers=1)

    sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
    
    mlp = MLP(
        in_sizes=h.input_sizes,
        out_sizes=h.output_sizes,
        activation=h.activation,
    ).to(device)
    
    steps = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=h.learning_rate)
    mlp.train()
    
    loss_list = []
    acc_list = []
    for epoch in range(0, a.training_epoch):
        print(f'Starting epoch {epoch}')
        for i, data in tqdm(enumerate(training_dataloader)):
            imgs, points, labels = data[IMAGE_KEY], data[POINT_KEY], data[LABEL_KEY]
            imgs = torch.autograd.Variable(imgs.to(device, non_blocking=True)) # [batchsize, 47, 47, 3]
            points = torch.autograd.Variable(points.to(device, non_blocking=True)) # [batchsize, 300, 3]
            labels = torch.autograd.Variable(labels.to(device, non_blocking=True)).squeeze(1)
            optimizer.zero_grad()
            flattened_imgs = torch.flatten(imgs, 1) # [batchsize, 6627]
            flattened_points = torch.flatten(points, 1) # [batchsize, 900]
            inputs = torch.cat((flattened_imgs, flattened_points), dim=1) # [batchsize, 7527]
            outputs = mlp(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # record
            pred = torch.argmax(outputs, dim=1)
            acc = accuracy(pred, labels)
            loss_list.append(loss.item())
            acc_list.append(acc)
            
            if steps % (a.summary_interval//2) == 0:
                sw.add_scalar("training/loss", sum(loss_list)/max(len(loss_list), 1), steps)
                sw.add_scalar("training/acc", sum(acc_list)/max(len(acc_list), 1), steps)
                loss_list = []
                acc_list = []

            if steps % a.summary_interval == 0:
                # validation
                mlp.eval()
                torch.cuda.empty_cache()
                val_loss = 0
                with torch.no_grad():
                    val_acc = []
                    for j, batch in enumerate(testing_dataloader):
                        imgs, points, labels = batch[IMAGE_KEY].to(device, non_blocking=True), batch[POINT_KEY].to(device, non_blocking=True), batch[LABEL_KEY].to(device, non_blocking=True).squeeze(1)
                        flattened_imgs = torch.flatten(imgs, 1) # [batchsize, 6627]
                        flattened_points = torch.flatten(points, 1) # [batchsize, 900]
                        inputs = torch.cat((flattened_imgs, flattened_points), dim=1) # [batchsize, 7527]
                        outputs = mlp(inputs)
                        loss = criterion(outputs, labels)
                        pred = torch.argmax(outputs, dim=1)
                        val_acc.append(accuracy(pred, labels))
                        val_loss += loss.item()

                    val_loss = val_loss / (j+1)
                    sw.add_scalar("validation/loss", val_loss, steps)
                    sw.add_scalar("validation/acc", np.mean(val_acc), steps)
                    
                mlp.train()
            optimizer.step()
            steps += 1

    # Process is complete.
    print('Training process has finished.')




if __name__ == '__main__':
    main()