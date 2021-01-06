# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:14:01 2019
reward
@author: ztm
"""

import os
import random
import math
import copy

import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


# 计算reward, 就是对于已经生成的不同长度的sentence，扩展到完整的句子，再用discriminator来打分
class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, update_rate):
        self.ori_model = model # G            # 参数不更新
        self.own_model = copy.deepcopy(model) # G.copy() 参数实时更新
        self.update_rate = update_rate # 更新频率 0.8

    def get_reward(self, x, num, discriminator):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : 16, roll-out number
            discriminator : discrimanator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num): # num = 16
            for l in range(1, seq_len): # given_num 如果给定20最后只到19
                data = x[:, 0:l]
                samples = self.own_model.sample(batch_size, seq_len, data) # 用G生成fake samples
                pred = discriminator(samples)
                pred = pred.cpu().data[:,1].numpy() # 得到每个句子的分值
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred # 把所有得分加在一起

            # for the last token 最后一个不需要进行采样，因为已经是完整的序列
            pred = discriminator(x) #最后一个word就不用自己run生成了，直接读取input_x
            pred = pred.cpu().data[:, 1].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len,再取平均
        return rewards # 这样，就得到了对于任意长度的句子的rewards

    def update_params(self):
        dic = {} # lstm.Wi ,ori
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data # ori
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'): # embedding weights 不用更新
                param.data = dic[name] # ori
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
                
                
                
                
                
                
                
                
                
                