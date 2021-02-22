# -*- coding: utf-8 -*-

import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import csv,os,time,pickle
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque

from load_subtrees import read_pair
from load_data import load_dataset, read_valid_file
from network import ContrastiveLoss, Siamese

def weights_init(mod):
    classname=mod.__class__.__name__
    print(classname)
    if classname.find('Conv')!= -1:   
        mod.weight.data.normal_(0.0,0.02)
    if classname.find('Linear')!= -1:    
        mod.weight.data.normal_(0.0,0.1)
    elif classname.find('BatchNorm')!= -1:
        mod.weight.data.normal_(1.0,0.02) 
        mod.bias.data.fill_(0)  

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func('EarlyStopping counter: ', self.counter, 'out of', self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func('Validation loss decreased ({',self.val_loss_min,':.6f} --> {',val_loss,':.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss        

def Iterator(intra_pair, inter_pair, batch_size, shape, _i):
    h = shape[0]; w = shape[1]; c = shape[2]

    while True:
        intra_batch_pairs = intra_pair[batch_size*_i:batch_size*(_i+1)]
        inter_batch_pairs = inter_pair[batch_size*_i:batch_size*(_i+1)]
        if(min(len(intra_batch_pairs),len(inter_batch_pairs)) < batch_size):
            _i = 0
            continue
        intra_batch = dirs_to_images_and_labels(intra_batch_pairs, h, w, c)
        inter_batch = dirs_to_images_and_labels(inter_batch_pairs, h, w, c)
        x1 = np.concatenate((intra_batch[0], inter_batch[0]), axis = 0)
        x2 = np.concatenate((intra_batch[1], inter_batch[1]), axis = 0)
        # norm
        x1 = x1/255
        x2 = x2/255   
        # set y(lables)
        y = np.empty((2*batch_size,),dtype="int")
        y[:batch_size] = 1
        y[batch_size:] = 0
        # shuffle
        index = [j for j in range(len(y))]
        random.shuffle(index)
        x1 = x1[index]
        x2 = x2[index]
        y = y[index]
        _i+=1
        yield ([x1,x2],y)
        
def dirs_to_images_and_labels(batch_dirs, height=128, width=128, channel=3, resize =0): 
    number_of_pairs = len(batch_dirs)
    image_pairs = [np.empty((number_of_pairs,height, width, channel),dtype="uint8") for i in range(2)]
    i = 0
    for dir_pair in batch_dirs:
        if(i >= number_of_pairs):
            break
        image1 = Image.open(dir_pair[0])
        if resize != 0:
            image1 = image1.resize((width,height),Image.ANTIALIAS)
        image1 = np.asarray(image1).astype(np.float64)
        image_pairs[0][i, :, :, :] = image1
        image2 = Image.open(dir_pair[1])
        if resize != 0:
            image2 = image2.resize((width,height),Image.ANTIALIAS)
        image2 = np.asarray(image2).astype(np.float64)
        image_pairs[1][i, :, :, :] = image2
        i += 1
    return image_pairs

if __name__ == '__main__':   
    dataset_path = r'.\p_app_Td_sts_resized' # subtree imgs
    input_shape = (256, 512, 3)
    learning_rate = 0.00005
    batch_size = 16
    epoch = 30
    #----------------------read data-----------------------------------------------------
    ui_dictionary = load_dataset(dataset_path)
    _file = r'.\data\data.txt'
    train_apps, valid_apps, test_apps = read_valid_file(_file)
    _train_pair_file = r'.\data\train_st_pair.txt'      
    train_intra_pair, train_inter_pair = read_pair(_train_pair_file) 
    train_size = min(len(train_intra_pair), len(train_inter_pair))        
    steps_per_epoch_train = train_size//batch_size     
    #----------------------build network-----------------------------------------------------
    net = Siamese()
    net.apply(weights_init)
    net.cuda()
    print(net)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate,weight_decay=0.95)
    criterion = ContrastiveLoss()
    train_loss = []
    early_stopping = EarlyStopping(patience=5, verbose=True)            
#    #----------------------train-----------------------------------------------------
    global _i
    for e in range(epoch): 
        print('epoch: ', e)
        time_start = time.time()
        loss_val = 0
        _i = 0
        train_g = Iterator(train_intra_pair, train_inter_pair,batch_size, input_shape, _i)
        for batch_id, ([x1,x2],y) in enumerate(train_g):
            if batch_id > steps_per_epoch_train:
                print(batch_id, 'loss_val: ',loss_val,'time: ', time.time() - time_start)
                break
            x1, x2, y = torch.FloatTensor(x1).cuda(), torch.FloatTensor(x2).cuda(), torch.FloatTensor(y).cuda()
            x1 = x1.permute(0,3,1,2); x2 = x2.permute(0,3,1,2)
            output1, output2 = net.forward(x1, x2)
            loss = criterion(output1, output2, y)    
            loss_val += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_id % 30 == 0:
                accuracy = []
                for idx, logit in enumerate([output1, output2]):
                    corrects = (torch.max(logit, 1)[1].data == y.long().data).sum()                   
                    accu = float(corrects) / float(y.size()[0])
                    accuracy.append(accu)     
                for idx, accu in enumerate(accuracy):
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}\tOut: {}\tAccu: {:.2f}'.format(
                        e, batch_id * len(y), 2*train_size,
                        100. * batch_id / steps_per_epoch_train, loss.item(), idx, accu * 100.))
                    
        torch.save(net.state_dict(), 'models_torch/' + 'torch_siamese-' + str(e) + ".pkl")                                
        train_loss.append(loss_val)
        
        early_stopping(loss_val, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
                
        with open('models_torch/'+'train_loss', 'wb') as f:
            pickle.dump(train_loss, f)
    torch.save(net.state_dict(), 'models_torch/' + 'torch_siamese.pkl')
