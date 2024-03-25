'''
Author: Suizhi HUANG && sunrisen.huang@gmail.com
Date: 2024-03-25 15:38:44
LastEditors: Suizhi HUANG && sunrisen.huang@gmail.com
LastEditTime: 2024-03-25 17:00:06
FilePath: /HPV/main.py
Description: 
Copyright (c) 2024 by $Suizhi HUANG, All Rights Reserved. 
'''

import datetime
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from data import get_data
from loss import BPRLoss
from model import DeepFM
from utils import *


def main(
    model,
    train_loader,
    valid_loader,
    epochs,
    device,
    logger,
    criterion,
    optimizer,
    scheduler,
    test_intervals=5,
):
    for _ in range(epochs):
        """训练部分"""
        model.train()
        print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        logger.info(
            "Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr'])
        )
        logger.info('Epoch: {}'.format(_ + 1))
        print('Epoch: {}'.format(_ + 1))
        train_loss_sum = 0.0
        start_time = time.time()
        for idx, x in enumerate(train_loader):
            cate_fea, nume_fea, label = x[0], x[1], x[2]
            cate_fea, nume_fea, label = (
                cate_fea.to(device),
                nume_fea.to(device),
                label.float().to(device),
            )
            pred = model(cate_fea, None).view(-1)

            # Find negative samples
            neg_indices = torch.where(label == 0)[0]
            neg_samples = pred[neg_indices]

            # Find positive samples
            pos_indices = torch.where(label == 1)[0]
            pos_samples = pred[pos_indices]

            # Randomly match negative samples with positive samples
            neg_indices_matched = torch.randint(
                0, len(pos_indices), size=(len(neg_indices),)
            )
            neg_samples_matched = pos_samples[neg_indices_matched]
            # replicate positive samples to match negative samples, while make sure the idx is in the range of negative samples
            pos_samples = pos_samples.repeat(len(neg_indices) // len(pos_indices) + 1)

            # Calculate loss
            loss = criterion(pos_samples, neg_samples_matched)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            if (idx + 1) % 50 == 0 or (idx + 1) == len(train_loader):
                logger.info(
                    "Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                        _ + 1,
                        idx + 1,
                        len(train_loader),
                        train_loss_sum / (idx + 1),
                        time.time() - start_time,
                    )
                )
        scheduler.step()

        if (_ + 1) % test_intervals == 0:
            evaluate(model, valid_loader, device, logger, path + "/best.pth")


def evaluate(model, valid_loader, device, logger, path):
    model.eval()
    best_sens = 0
    with torch.no_grad():
        valid_labels, valid_preds = [], []
        for idx, x in tqdm(enumerate(valid_loader)):
            cate_fea, nume_fea, label = x[0], x[1], x[2]
            cate_fea, nume_fea = cate_fea.to(device), nume_fea.to(device)
            pred = model(cate_fea, nume_fea).reshape(-1).data.cpu().numpy().tolist()
            valid_preds.extend(pred)
            valid_labels.extend(label.cpu().numpy().tolist())
    accuracy = accuracy_score(valid_labels, valid_preds)
    sensitivity_score = recall_score(valid_labels, valid_preds)
    specificity_score = recall_score(valid_labels, valid_preds, pos_label=0)
    if sensitivity_score > best_sens:
        best_sens = sensitivity_score
        torch.save(model.state_dict(), path)
    logger.info(
        'Current sensitivity: %.6f, Best sensitivity: %.6f\n'
        % (sensitivity_score, best_sens)
    )
    logger.info('Current acc: %.6f' % (accuracy))
    logger.info('Current specificity: %.6f' % (specificity_score))


def set_seed(seed=2024):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = parse_opts()
    set_seed()

    data, dense_features, sparse_features = get_data(
        args.train_file, args.sparse_feature_num, args.dense_feature_num
    )

    train, valid = train_test_split(data, test_size=0.2, random_state=2020)
    print(train.shape, valid.shape)

    train_dataset = Data.TensorDataset(
        torch.LongTensor(train[sparse_features].values),
        torch.FloatTensor(train[dense_features].values),
        torch.FloatTensor(train['label'].values),
    )
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    valid_dataset = Data.TensorDataset(
        torch.LongTensor(valid[sparse_features].values),
        torch.FloatTensor(valid[dense_features].values),
        torch.FloatTensor(valid['label'].values),
    )
    valid_loader = Data.DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    cate_fea_nuniqs = [data[f].nunique() for f in sparse_features]
    model = DeepFM(cate_fea_nuniqs, nume_fea_size=len(dense_features))
    model.to(device)

    criterion = BPRLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    path = os.path.join("./exp/", args.exp)
    os.makedirs(path, exist_ok=True)
    logger = get_logger(path)
    logger.info('Start training ...')
    main(
        model,
        train_loader,
        valid_loader,
        args.n_epochs,
        device,
        logger,
        criterion,
        optimizer,
        scheduler,
        args.test_intervals,
    )
