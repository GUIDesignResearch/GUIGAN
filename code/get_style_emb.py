# -*- coding: utf-8 -*-
"""
save embs in file
"""
import torch
import torch.nn as nn
import csv,os,sys,time
import numpy as np
from PIL import Image
from apted import APTED, Config
from  apted.helpers import Tree

sys.path.append(r'.\StyleEmbedding')
from load_data import get_s_app
from network import Siamese

def get_images(train_uis, input_shape):
    num = len(train_uis)
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
                x_info.append([ui_id+'_'+k1,t.split(':')[1].strip(),t.split(':')[2].strip().split('_')[0],t.split(':')[2].strip().split('_')[1],tui,t.split(':')[3].strip(),])
                break
    return x_info

if (__name__ == "__main__"):
    
    input_shape = (256, 512, 3)
    net = Siamese()
    net.cuda()
    net.load_state_dict(torch.load('StyleEmbedding/models_torch/torch_siamese-epoch.pkl'))
    net.eval()    
    c_size = 50

    file_csv = r'app_details.csv'
    txt_dir  = r'.\aTrees_dict_app'    
    st_dir   = r'.\p_app_Td_sts'
    cd_img   = r'.\p_app_Td_sts_resized'
    path_file_name = r'.\data\categories_app_emb'
    
    appsl, appsd = get_s_app(file_csv, st_dir)
    
    for (cat,cat_apps) in appsd.items():
        print('\ncategory: ', cat)
        print('\ncategory_apps: ', cat_apps)
        train_uis = []
        for app in os.listdir(cd_img):
            if app in cat_apps:
                app_dir = os.path.join(cd_img,app)
                for ui in os.listdir(app_dir):
                    ui_dir = os.path.join(app_dir,ui)
                    train_uis.append(ui_dir)  
               
        c_num = len(train_uis)/c_size 
        x_train_embedding = []
        if(c_num>int(c_num)):
            c_num = int(c_num) + 1
        else:
            c_num = int(c_num)
        for i in range(c_num):
            
            x_train_batch = train_uis[c_size*i:c_size*(i+1)]
            x_train_batch = get_images(x_train_batch, input_shape)            
            x_train_batch = torch.FloatTensor(x_train_batch).cuda()
            x_train_batch = x_train_batch.permute(0,3,1,2)
            
            _output = net.forward_one(x_train_batch) # (c, 2)
            _output = _output.reshape(_output.shape[0], -1)
            emb = _output.detach().cpu().numpy()
            
            if i == 0:
                x_train_embedding = emb
            else:
                x_train_embedding = np.concatenate((x_train_embedding,emb),axis=0)
        
        path_file_name = os.path.join(path_file_name, str(cat)+'.txt')
        with open(path_file_name, "a") as f:
            for i in range(len(train_uis)):
                ui = str(train_uis[i])
                embs = x_train_embedding[i]
                f.write(ui+'::')
                [f.write(str(embs[j])+',') for j in range(len(embs))]
                f.write(';\n')                
                
                