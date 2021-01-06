# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:12:59 2019
D
@author: ztm
"""

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """A CNN for text classification
    Discriminator是一个具有highway的CNN
    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    num_classes:  2    # 二分类
    vocab_size:   5000 # 词向量个数
    emb_dim：     64  embedding维度, 词向量维度 ,lstm_input_size
    filter_sizes: [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  15,  20]
    num_filters:  [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]  
    d_dropout:    0.75
    """
    def __init__(self, num_classes, vocab_size, emb_dim, filter_sizes, num_filters, dropout):
        super(Discriminator, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(sum(num_filters), num_classes)
        self.softmax = nn.LogSoftmax()
        self.init_parameters()

    def forward(self, x):
        """
        Args:
            x:    (batch_size * seq_len)       16 * 20
            pred: (batch_size * num_classes)   16 * 2
                        
            squeeze: 
            (1)squeeze(a)就是将a中所有为1的维度删掉。不为1的维度没有影响。
            (2)a.squeeze(N) 就是去掉a中指定的维数为一的维度。
            (3)还有一种形式就是b=torch.squeeze(a，N) a中去掉指定的定的维数为一的维度。
        """
        emb = self.emb(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) *  F.relu(highway) + (1. - torch.sigmoid(highway)) * pred
        pred = self.softmax(self.lin(self.dropout(pred)))
#        print('D_pred.shape', pred.shape)
        return pred
    
    

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)