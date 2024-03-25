'''
Author: Suizhi HUANG && sunrisen.huang@gmail.com
Date: 2024-03-25 15:41:40
LastEditors: Suizhi HUANG && sunrisen.huang@gmail.com
LastEditTime: 2024-03-25 19:29:13
FilePath: /HPV/data.py
Description: 
Copyright (c) 2024 by $Suizhi HUANG, All Rights Reserved. 
'''

import datetime
import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def get_data(file_path, sparse_num, dense_num, balance):
    data = pd.read_csv(file_path)

    sparse_features = ['C' + str(i + 1) for i in range(sparse_num)]
    dense_features = ['I' + str(i + 1) for i in range(dense_num)]

    data[sparse_features] = data[sparse_features].fillna(
        '-10086',
    )
    data[dense_features] = data[dense_features].fillna(
        0,
    )
    data.apply(pd.to_numeric, errors="ignore")
    data = data.sample(frac=1)
    target = ['label']

    ## 类别特征labelencoder
    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    ## 数值特征标准化
    for feat in tqdm(dense_features):
        mean = data[feat].mean()
        std = data[feat].std()
        data[feat] = (data[feat] - mean) / (std + 1e-12)  # 防止除零

    if balance > 0:
        print(type(data))
        data = balance_data(data, balance)
        print(type(data))

    return data, dense_features, sparse_features


def balance_data(data, balance):
    positive_samples = data[data['label'] == 1]
    num_positive_samples = len(positive_samples)
    negative_samples = data[data['label'] == 0]
    num_negative_samples = len(negative_samples)

    unit = min(num_positive_samples, num_negative_samples)
    # num_positive_samples = balance * unit
    num_negative_samples = balance * unit

    # # Replicate positive samples to balance the number of positive and negative samples
    positive_samples = positive_samples.sample(n=num_positive_samples, replace=True)
    # Reduce the number of negative samples to match the number of positive samples
    negative_samples = negative_samples.sample(n=num_negative_samples)

    # Combine the replicated positive samples and reduced negative samples
    balanced_data = pd.concat([positive_samples, negative_samples])
    # Shuffle the data
    balanced_data = balanced_data.sample(frac=1)

    return balanced_data
