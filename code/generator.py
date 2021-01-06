# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:11:53 2019
G
@author: ztm
"""

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class Generator(nn.Module): # 生成长度为20的句子
    """Generator 
    num_emb：    5000 # 词向量个数
    emb_dim：    32  embedding维度, 词向量维度 ,lstm_input_size
    hidden_dim： 32  中间层特征维度
    """
    def __init__(self, num_emb, emb_dim, hidden_dim, use_cuda):
        super(Generator, self).__init__()
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.emb = nn.Embedding(num_emb, emb_dim) # subtree由于不重复，输入num_emb会很大，造成Embedding特别稀疏,emb_dim为256
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True) # num_layers Default: 1
        self.lin = nn.Linear(hidden_dim, num_emb)
        self.softmax = nn.LogSoftmax()
        self.init_params()

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len), sequence of tokens generated by generator
        """
#        print('foward input: ', x.shape)
        emb = self.emb(x)
#        print('foward emb: ', emb.shape)
        h0, c0 = self.init_hidden(x.size(0))
#        print('foward h0: ', h0.shape)
#        print('foward c0: ', c0.shape)
        output, (h, c) = self.lstm(emb, (h0, c0)) # shape=()
#        print('foward output: ', output.shape)
        pred = self.softmax(self.lin(output.contiguous().view(-1, self.hidden_dim))) # view类似reshape,且跟contiguous联合使用
#        pred = self.softmax(self.lin(output.reshape(-1, self.hidden_dim)))
#        print('foward pred: ', pred.shape)
        return pred

    def step(self, x, h, c):
        """
        Args:
            x: (batch_size,  1), sequence of tokens generated by generator
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """
#        print('x.shape: ', x.shape) # torch.Size([32, 1])
        emb = self.emb(x) # torch.Size([32, 1, 32])
#        print('emb.shape: ', emb.shape)
        output, (h, c) = self.lstm(emb, (h, c))
        pred = F.softmax(self.lin(output.view(-1, self.hidden_dim)), dim=1)
        return pred, h, c

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
            
    def sample(self, batch_size, seq_len, x=None):
#        res = []
        flag = False # whether sample from zero
        if x is None: # 从zero开始
            flag = True
        if flag:
            x = Variable(torch.zeros((batch_size, 1)).long()) # x与target_lstm一样
#            print('\nx_start: ', x.shape)
#            print(x)
        if self.use_cuda:
            x = x.cuda()
        h, c = self.init_hidden(batch_size) # x与target_lstm一样
        samples = []
#        H = 0
        if flag: # 从零开始
            # 该为 height(1:T) < H 
            for i in range(seq_len): # 句子长度
                output, h, c = self.step(x, h, c)
#                print('\nx.shape',output.shape)
                x = output.multinomial(1) # one-hot --> argmax
#                print('\nx.shape',x.shape, x)
#                print('samples: ', samples)
                samples.append(x)
        else:
#            print('not flag!')
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            for i in range(given_len):
                output, h, c = self.step(lis[i], h, c)
                samples.append(lis[i])
            x = output.multinomial(1)
            for i in range(given_len, seq_len):
                samples.append(x)
                output, h, c = self.step(x, h, c)
                x = output.multinomial(1)
                
        output = torch.cat(samples, dim=1)
        return output

if __name__ == '__main__':        
    # Genrator Parameters
    g_emb_dim = 32      # Embedding 维度
    g_hidden_dim = 32   # nn.LSTM(emb_dim, hidden_dim)
    #g_sequence_len = 10 # 长度为20的句子
    #g_sequence_len = 15 
    g_sequence_len = 20
    BATCH_SIZE = 32
    VOCAB_SIZE = 1000
    
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, 'True')   
    generator = generator.cuda()
    samples = generator.sample(BATCH_SIZE, g_sequence_len)
    print('\n')
    zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
    if samples.is_cuda:
        zeros = zeros.cuda()
    inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
    targets = Variable(samples.data).contiguous().view((-1,))
    prob = generator.forward(inputs)
    prob1 = generator(inputs)
    print('inputs.shape: ', inputs.shape) # torch.Size([32, 20])
    i = inputs.shape[1:]
    print('i ', i)
    from torchsummary import summary
    from torchviz import make_dot
    summary(generator, input_size=(32, 20))    
    m1 = make_dot(prob, params=dict(generator.named_parameters()))
    m1.render('SeqGAN_g', view=False)
    
    
    
    
    
    
    
    
    
    
    
    