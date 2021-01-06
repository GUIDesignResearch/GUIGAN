# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:18:06 2019
(9_1)save embs in txt file by categories
@author: ztm
"""

import csv,os,gzip,sys,pickle,gc,time
import random
import numpy as np
from PIL import Image
from collections import Counter
import keras.backend as K
from keras import callbacks
from keras.utils import plot_model
import matplotlib.pyplot as plt
from concurrent import futures

from apted import APTED, Config
from  apted.helpers import Tree

import sys
sys.path.append(r'D:\SpyderWorkSpace\GUI_Ganerator')
from load_subtrees import read_pair
from load_data import load_dataset, read_valid_file, get_s_app
from model import siamese_net


def load_gz(file1):
    with gzip.open(file1,'rb') as fil:
        ds = []
        news_ids = []
        d_list = pickle.load(fil)   
        d1 = Counter(d_list)
        
        for id in d_list:            
            if id not in news_ids:  # new
                news_ids.append(id)
                if(d1[id] != 1):
                    print('new id:', id, '并重复',d1[id], '次')
                    count = 1
                    _data = pickle.load(fil)
                else:
                    print('new id and 不重复:', id)
                    _data = pickle.load(fil)
                    ds.append(_data)
            else:                   # old
                print('old id:', id)
                _data = np.concatenate((_data,pickle.load(fil)),axis = 0)
                count+=1
                print('count:',count)
                if(count == d1[id]):
                    ds.append(_data)
        return ds

def get_images(train_uis, input_shape):
    num = len(train_uis)
#    print('Tensor length:', num)
    height, width, channel = input_shape[0], input_shape[1], input_shape[2]   
    images = np.empty((num,height, width, channel),dtype="uint8")
    i = 0
    for dir_pair in train_uis:
        if(i >= num):
            break
        image1 = Image.open(dir_pair)
        image1 = np.asarray(image1).astype(np.float64)
        images[i, :, :, :] = image1
        i += 1
    return images

def get_3data(train_intra_pair, train_inter_pair):
    train_uis = []
    train_apps_name = []
    train_UI_num = []
    for pair in train_intra_pair:
        if(pair[0] not in train_uis):
            train_uis.append(pair[0])
            train_apps_name.append(os.path.split(os.path.split(pair[0])[0])[1])
            train_UI_num.append(os.path.splitext(os.path.basename(pair[0]))[0])
        if(pair[1] not in train_uis):
            train_uis.append(pair[1])    
            train_apps_name.append(os.path.split(os.path.split(pair[1])[0])[1])
            train_UI_num.append(os.path.splitext(os.path.basename(pair[1]))[0])
    for pair in train_inter_pair:
        if(pair[0] not in train_uis):
            train_uis.append(pair[0])
            train_apps_name.append(os.path.split(os.path.split(pair[0])[0])[1])
            train_UI_num.append(os.path.splitext(os.path.basename(pair[0]))[0])
        if(pair[1] not in train_uis):
            train_uis.append(pair[1])
            train_apps_name.append(os.path.split(os.path.split(pair[1])[0])[1])
            train_UI_num.append(os.path.splitext(os.path.basename(pair[1]))[0])
    return train_uis, train_apps_name, train_UI_num

def load_emb_info(x):
    x_id = x.split('::')[0]
    x_em = x.split('::')[1].split(',')
    x_em.remove(x_em[-1])
    x_em = [[float(e) for e in x_em]]
    return x_id,x_em

def read_data(path_file_name):
    print('\nloading data')
    starttime = time.time()
    x_train_embedding = []
    x_ids = []
    with open(path_file_name, "r") as f:
        f_content = f.read()
        x_train_embs = f_content.split(';\n')
        x_train_embs.remove(x_train_embs[-1])
        for x in x_train_embs:
            x_id = x.split('::')[0]
            x_ids.append(x_id)
            x_em = x.split('::')[1].split(',')
            x_em.remove(x_em[-1])
            x_em = [[float(e) for e in x_em]]
            if x_train_embedding ==[]:
                x_train_embedding = x_em
            else:                
                x_train_embedding = np.concatenate((x_train_embedding, x_em), axis = 0)
    endtime = time.time()
    dtime = endtime - starttime
    print("\nLoading time for the SubTrees in training data：%.8s s" % dtime)
    return x_ids,x_train_embedding

def get_ui_info(train_uis,txt_dir):
    x_info = []
    for tui in train_uis:
        tui_dir = os.path.split(tui)
        tapp = os.path.split(tui_dir[0])[1]
        tui_dir1 = os.path.splitext(tui_dir[1])
        ui_id = tui_dir1[0].split('_')[0]
        subtree_id = tui_dir1[0].split(ui_id+'_')[-1]
        s = os.path.join(tapp,ui_id)
        ui_tree_dir = os.path.join(txt_dir,s+'.txt')
        f2 = open(ui_tree_dir,'r')
        f_content = f2.read() 
        aTrees2 = f_content.split(',\n')
        aTrees2.remove(aTrees2[-1])
        for t in aTrees2:
            k1 = t.split(':')[0].strip()
            if k1 ==subtree_id:
#                x_info.append([ui_id+'_'+k1,t.split(':')[1].strip(),t.split(':')[2].strip().split('_')[0],t.split(':')[2].strip().split('_')[1],tui])
                x_info.append([ui_id+'_'+k1,t.split(':')[1].strip(),t.split(':')[2].strip().split('_')[0],t.split(':')[2].strip().split('_')[1],tui,t.split(':')[3].strip(),])
                break
    return x_info
#=======================start======================================================================
if (__name__ == "__main__"):
    
    input_shape = (256, 512, 3)
    learning_rate = 10e-4
    batch_size = 16
    epoch = 50
        
    # 构建网络，
    l2_penalization = {}
    l2_penalization['Conv1'] = 1e-2
    l2_penalization['Conv2'] = 1e-2
    l2_penalization['Conv3'] = 1e-2
    l2_penalization['Conv4'] = 1e-2
    l2_penalization['Dense1'] = 1e-4
    # L2 alone
    siamese_network, cnn = siamese_net(input_shape, l2_penalization, learning_rate)
#    cnnh5 = r'D:\SpyderWorkSpace\GUI_Ganerator\models\trained_cnn.h5'
    cnn.load_weights(r'D:\SpyderWorkSpace\GUI_Ganerator\models\trained_cnn.h5')
    
    c_size=200
    layer_output = K.function([cnn.layers[0].input],[cnn.get_layer('Dense1').output])
    
    cd_img = r'D:\zhu\chen1\data\pick8\p_app_resize_Td_sts_c_resize512_noui1'
    txt_dir = r'D:\zhu\chen1\data\pick8\aTrees_dict_app'
    file_csv = r'F:\2017\zhu\RicoDataset\app_details.csv'
    st_dir = r'D:\zhu\chen1\data\pick8\p_app_resize_Td_sts'
    
    appsl, appsd = get_s_app(file_csv, st_dir) # 按照category中app的数量
    
    for (cat,cat_apps) in appsd.items():
        print('\ncategory: ', cat)
        print('\ncategory_apps: ', cat_apps)
        train_uis = []
        for app in os.listdir(cd_img):
            if app in cat_apps:
                app_dir = os.path.join(cd_img,app)
                for ui in os.listdir(app_dir):
    #                print(app,ui)
                    ui_dir = os.path.join(app_dir,ui)
                    train_uis.append(ui_dir)  # 需要embedding的subtrees
               
        # 输入数据，生成embedding
        c_num = len(train_uis)/c_size # 迭代次数
        x_train_embedding = []
        if(c_num>int(c_num)):
            c_num = int(c_num) + 1
        else:
            c_num = int(c_num)
        for i in range(c_num):
            x_train_batch = train_uis[c_size*i:c_size*(i+1)]
            x_train_batch = get_images(x_train_batch, input_shape)
            _output =layer_output([x_train_batch])[0]
            o1 = _output.reshape(_output.shape[0], -1)
            
            if(i==0):
                x_train_embedding = o1
            else:
                x_train_embedding = np.concatenate((x_train_embedding, o1), axis = 0)               
                
#        # 把x_train_embedding写成文件    
##        path_file_name = r'D:\SpyderWorkSpace\GUI_Ganerator\data\x_train_emb.txt' # 5781.519 s
#    #    path_file_name = r'D:\SpyderWorkSpace\GUI_Ganerator\data\x_test_emb.txt'  # 86.30812 s
#        
#        path_file_name = r'D:\SpyderWorkSpace\GUI_Ganerator\data\categories_app_emb'
#        path_file_name = os.path.join(path_file_name, str(cat)+'.txt')
#        
#        # 保存embedding到txt文件
#        with open(path_file_name, "a") as f:
#            for i in range(len(train_uis)):
#                ui = str(train_uis[i])
#                embs = x_train_embedding[i]
#                f.write(ui+'::')
#                [f.write(str(embs[j])+',') for j in range(len(embs))]
#                f.write(';\n')
    
    
    
        
    