# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:41:01 2019

@author: ASUS
"""

import os
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda, Dropout, BatchNormalization,\
Activation,GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
import keras.backend as K
from keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
os.environ['CUDA_VISIBLE_DEVICES']= '-1'
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存 
session = tf.Session(config=config)
session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True  # 设置allow_grouth，根据运行时的需要来分配GPU内存
sess = tf.Session(config=session_conf)

def siamese_net(input_shape, l2_regl_penalization, learning_rate, dr=False):
    cn = Sequential()
    
    # L2 Approach
    cn.add(Conv2D(32, (7, 7), input_shape= input_shape, kernel_regularizer=l2(l2_regl_penalization['Conv1']), name='Conv1'))
    cn.add(Activation('relu'))
    cn.add(MaxPool2D())
    cn.add(Conv2D(64, (5, 5), kernel_regularizer=l2(l2_regl_penalization['Conv2']), name='Conv2'))
    cn.add(Activation('relu'))
    cn.add(MaxPool2D())
    cn.add(Conv2D(128, (3, 3), kernel_regularizer=l2(l2_regl_penalization['Conv3']), name='Conv3'))
    cn.add(Activation('relu'))
    cn.add(MaxPool2D())
    cn.add(Conv2D(256, (3, 3), kernel_regularizer=l2(l2_regl_penalization['Conv4']), name='Conv4'))
    cn.add(Activation('relu'))
    cn.add(MaxPool2D())
    cn.add(Flatten())
    
#    # 跟AE一样，输入很小的
#    cn.add(Conv2D(32, (3, 3), input_shape= input_shape,
#                  kernel_regularizer=l2(l2_regl_penalization['Conv1']), name='Conv1'))
#    cn.add(Activation('relu'))
#    cn.add(MaxPool2D())
#    cn.add(Conv2D(64, (3, 3), kernel_regularizer=l2(l2_regl_penalization['Conv2']), name='Conv2'))
#    cn.add(Activation('relu'))
#    cn.add(MaxPool2D())
#    cn.add(Conv2D(128, (3, 3), kernel_regularizer=l2(l2_regl_penalization['Conv3']), name='Conv3'))
#    cn.add(Activation('relu'))
#    cn.add(MaxPool2D())
#    cn.add(Conv2D(256, (3, 3), kernel_regularizer=l2(l2_regl_penalization['Conv4']), name='Conv4'))
#    cn.add(Activation('relu'))
#    cn.add(MaxPool2D())
#    cn.add(Flatten()) 
    
#    cn.add(Dense(128, kernel_regularizer=l2(l2_regl_penalization['Dense1']), name='Dense1')) # 输出维度
    cn.add(Dense(256, kernel_regularizer=l2(l2_regl_penalization['Dense1']), name='Dense1'))
#    cn.add(Dense(512, kernel_regularizer=l2(l2_regl_penalization['Dense1']), name='Dense1')) # 输出维度
#    cn.add(Dense(768, kernel_regularizer=l2(l2_regl_penalization['Dense1']), name='Dense1')) # 输出维度
#    cn.add(Dense(1024, kernel_regularizer=l2(l2_regl_penalization['Dense1']), name='Dense1')) # 输出维度
    if(not dr):
        print('no dropout')
        cn.add(Activation('sigmoid', name='output'))
    else:
        print('with dropout')
        cn.add(Activation('sigmoid', name='Dense2'))
        cn.add(Dropout(0.5, name='output'))

    # The pairs of images
    input_image_1 = Input(input_shape)
    input_image_2 = Input(input_shape)
    
    encoded_image_1 = cn(input_image_1)
    encoded_image_2 = cn(input_image_2)

#    plot_model(cn, to_file='./result/convolutional_net.png', show_shapes=True, show_layer_names=True)
    # L1 distance layer between the two encoded outputs
    # One could use Subtract from Keras, but we want the absolute value
    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]), name='l1_distance')
    l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])

    # Same class or not prediction
    prediction = Dense(units=1, activation='sigmoid')(l1_distance)
    model = Model(inputs=[input_image_1, input_image_2], outputs=prediction)

    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer=optimizer)
    return model, cn