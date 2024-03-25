'''
Author: Suizhi HUANG && sunrisen.huang@gmail.com
Date: 2024-03-25 16:01:06
LastEditors: Suizhi HUANG && sunrisen.huang@gmail.com
LastEditTime: 2024-03-25 16:23:17
FilePath: /HPV/loss.py
Description: 
Copyright (c) 2024 by $Suizhi HUANG, All Rights Reserved. 
'''

import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_scores, neg_scores):
        loss = torch.mean(-torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss
