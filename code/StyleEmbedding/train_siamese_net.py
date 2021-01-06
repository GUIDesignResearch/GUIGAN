# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:35:36 2019
(8)
@author: ztm
"""

import csv,os
import random
import numpy as np
from PIL import Image
#import keras.backend as K
from keras import callbacks
from keras.utils import plot_model
import matplotlib.pyplot as plt

from load_subtrees import read_pair
from load_data import load_dataset, read_valid_file
from model import siamese_net

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session 
# 指定第一块GPU可用 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #指定GPU的第二种方法 
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.5 #定量
config.gpu_options.allow_growth = True  #按需
set_session(tf.Session(config=config)) 
print(tf.test.is_gpu_available())

def generator(intra_pair, inter_pair, batch_size, shape, _i):
    h = shape[0]; w = shape[1]; c = shape[2]
    # _i = 0
    # global i
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
#        nor1 = np.max(x1)
#        x1 = x1/nor1
#        nor2 = np.max(x2)
#        x2 = x2/nor2        
        # set the y(lables)
        y = np.empty((2*batch_size,),dtype="int")
        y[:batch_size] = 0
        y[batch_size:] = 1     
        # shuffle the data
        index = [j for j in range(len(y))]
        random.shuffle(index)
        x1 = x1[index]
        x2 = x2[index]
        y = y[index]
        _i+=1
        yield ([x1,x2],y)

def dirs_to_images_and_labels(batch_dirs, height=325, width=180, channel=3): # height原来360，裁剪后325
    number_of_pairs = len(batch_dirs)
    if(channel ==1):
        image_pairs = [np.empty((number_of_pairs,height, width, channel),dtype="float32") for i in range(2)]
        i = 0
        for dir_pair in batch_dirs:
            if(i >= number_of_pairs):
                break
            image1 = Image.open(dir_pair[0])             
            image1 = np.asarray(image1).astype(np.float64)
            image1 = image1[:, :, np.newaxis] # 手动加一个维度
#            print(image1.shape)
            image_pairs[0][i, :, :, :] = image1
            image2 = Image.open(dir_pair[1])
            image2 = np.asarray(image2).astype(np.float64)
            image2 = image2[:, :, np.newaxis]
            image_pairs[1][i, :, :, :] = image2
            i += 1
    else: # 3通道
        image_pairs = [np.empty((number_of_pairs,height, width, channel),dtype="uint8") for i in range(2)]
        i = 0
        for dir_pair in batch_dirs:
            if(i >= number_of_pairs):
                break
            image1 = Image.open(dir_pair[0])             
            image1 = np.asarray(image1).astype(np.float64)
#            print(image1.shape)
            image_pairs[0][i, :, :, :] = image1
            image2 = Image.open(dir_pair[1])
            image2 = np.asarray(image2).astype(np.float64)
            image_pairs[1][i, :, :, :] = image2
            i += 1
    return image_pairs  

def get_data_from_list(intra_pair, inter_pair, shape):
    h = shape[0]; w = shape[1]; c = shape[2]
    size = min(len(intra_pair), len(inter_pair))
    intra_pair = intra_pair[:size]; inter_pair = inter_pair[:size] # 数据找齐
    intra_batch = dirs_to_images_and_labels(intra_pair, h, w, c)
    inter_batch = dirs_to_images_and_labels(inter_pair, h, w, c)
    x1 = np.concatenate((intra_batch[0], inter_batch[0]), axis = 0)
    x2 = np.concatenate((intra_batch[1], inter_batch[1]), axis = 0)
    # norm
    nor1 = np.max(x1)
    x1 = x1/nor1
    nor2 = np.max(x2)
    x2 = x2/nor2        
    # set the y(lables)
    y = np.empty((2*size,),dtype="int")
    y[:size] = 0 # same is 0
    y[size:] = 1 # different is 1
    # shuffle the data
    index = [j for j in range(len(y))]
    random.shuffle(index)
    x1 = x1[index]
    x2 = x2[index]
    y = y[index]
    
#    x1 = np.array(x1)[index]
#    x2 = np.array(x2)[index]
#    y = np.array(y)[index]
    return [x1,x2],y   

def train1(model, cn, epoch, batch_size, intra_pair, inter_pair, valid_intra_pair, valid_inter_pair, shape, csv_name='log.csv'):
    train_size = min(len(train_intra_pair), len(train_inter_pair))
    x_valid, y_valid = get_data_from_list(valid_intra_pair, valid_inter_pair, shape)
    early_stop = callbacks.EarlyStopping(monitor='loss',patience=10)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.95 ** epoch))
    log = callbacks.CSVLogger(csv_name)
    siamese_network.fit_generator(generator(intra_pair, inter_pair, batch_size, shape),steps_per_epoch=train_size//batch_size, 
                    epochs=epoch,
                    validation_data=(x_valid, y_valid),
                    callbacks=[log,early_stop,lr_decay])
    siamese_network.save_weights('models/' + 'trained_siamese.h5')
    cn.save_weights('models/' + 'trained_cnn.h5')

def test1(model, batch_size, intra_pair, inter_pair, shape, log_file):    
    x_test, y_test = get_data_from_list(intra_pair, inter_pair, shape)
##    y_pred = model.predict(x_test)
    print('loaded testing data')
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test acc:', test_accuracy)
    print('Test loss:', test_loss)


#=======================start======================================================================
if (__name__ == "__main__"):
    T = True
#    T = False
    
    dataset_path = r'.\p_app_resize_Td_sts'
    
    input_shape = (256, 512, 3)
    learning_rate = 10e-4
    batch_size = 16
    epoch = 50
    
    # load the data 
#    ui_dictionary = load_dataset(dataset_path)
    _file = r'.\data\data.txt'
    train_apps, valid_apps, test_apps = read_valid_file(_file)
    
    _train_pair_file = r'.\data\train_st_pair.txt'
    _valid_pair_file = r'.\data\valid_st_pair.txt'
    _test_pair_file = r'.\data\test_st_pair.txt'  
    
    # l2-regularization penalization for each layer
    l2_penalization = {}
    l2_penalization['Conv1'] = 1e-2
    l2_penalization['Conv2'] = 1e-2
    l2_penalization['Conv3'] = 1e-2
    l2_penalization['Conv4'] = 1e-2
    l2_penalization['Dense1'] = 1e-4
    # Path where the logs will be saved
    tensorboard_log_path = './logs/siamese_net_lr10e-4'
        
    # L2 alone
    siamese_network, cnn = siamese_net(input_shape, l2_penalization, learning_rate)
    
    if(T):           
        train_intra_pair, train_inter_pair = read_pair(_train_pair_file)
        valid_intra_pair, valid_inter_pair = read_pair(_valid_pair_file)
        test_intra_pair, test_inter_pair = read_pair(_test_pair_file)
        print('===============Training===============')
        train1(siamese_network, cnn, epoch, batch_size, train_intra_pair, train_inter_pair, valid_intra_pair, valid_inter_pair, input_shape)

    else:
        print('===============Testing===============')
        test_intra_pair, test_inter_pair = read_pair(_test_pair_file)
        cnn.load_weights('models/' + 'trained_cnn.h5')
        siamese_network.load_weights('models/' + 'trained_siamese.h5')    
        test1(siamese_network, batch_size, test_intra_pair, test_inter_pair, input_shape, log_file)