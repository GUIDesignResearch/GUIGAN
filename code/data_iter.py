# -*- coding: utf-8 -*-

import os
import random
import math
import numpy as np
import torch

class DisDataIter(object): # pairs
    """ Toy data iter to load digits"""
    def __init__(self, real_data, fake_data_file, batch_size):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        real_data_lis = real_data
        fake_data_lis = self.read_file(fake_data_file)
        self.data = real_data_lis + fake_data_lis
        self.labels = [1 for _ in range(len(real_data_lis))] +\
                        [0 for _ in range(len(fake_data_lis))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        data = torch.LongTensor(np.asarray(data, dtype='int64'))
        label = torch.LongTensor(np.asarray(label, dtype='int64'))
        self.idx += self.batch_size
        return data, label

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis
    
    