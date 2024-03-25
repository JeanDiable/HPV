'''
Author: Suizhi HUANG && sunrisen.huang@gmail.com
Date: 2024-03-25 15:43:05
LastEditors: Suizhi HUANG && sunrisen.huang@gmail.com
LastEditTime: 2024-03-25 16:16:08
FilePath: /HPV/utils.py
Description: 
Copyright (c) 2024 by $Suizhi HUANG, All Rights Reserved. 
'''

import argparse
import time

from sklearn.metrics import recall_score


def parse_opts():
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sparse_feature_num', default=62, type=int, help='Number of sparse features'
    )
    parser.add_argument(
        '--dense_feature_num', default=0, type=int, help='Number of dense features'
    )
    parser.add_argument(
        '--train_file',
        default='./data/shuffled_hpv231229.csv',
        type=str,
        help='Training data.',
    )
    parser.add_argument(
        '--exp',
        default=f'{timestamp}',
        type=str,
        help='Experiment output directory.',
    )
    parser.add_argument('--num_workers', default=4, type=int, help='Number of jobs')

    parser.add_argument(
        '--learning_rate',
        default=0.01,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)',
    )
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument(
        '--n_epochs', default=50, type=int, help='Number of total epochs to run'
    )
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer.')
    parser.add_argument(
        '--test_intervals', default=5, type=int, help='Iteration for testing model'
    )

    args = parser.parse_args()

    return args


# 打印模型参数
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def sensitivity_score(y_pred, y):
    sensitivity = recall_score(y, y_pred)


def specificity_score(y_pred, y):
    specificity = recall_score(y, y_pred, pos_label=0)
