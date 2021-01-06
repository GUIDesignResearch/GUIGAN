# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:09:15 2019
(11_2)按照categories, 把start_list分开出来
@author: ztm
"""

import os,time,gc,random,math
import argparse
import tqdm
import numpy as np
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchviz import make_dot

from sklearn import (manifold, datasets, decomposition, ensemble,random_projection, metrics)
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import kneighbors_graph
from scipy.stats import mode

from apted import APTED, Config
from  apted.helpers import Tree

from generator import Generator
from discriminator import Discriminator
#from target_lstm import TargetLSTM
from reward import Rollout
from data_iter import GenDataIter1, DisDataIter,DisDataIter1,DisDataIter2

import sys
sys.path.append(r'D:\SpyderWorkSpace\GUI_Ganerator')
from load_data import read_valid_file, get_s_app
from load_style_emb import read_data,get_ui_info,get_3data
from load_subtrees import read_pair
from Find_opt_seg_T import get_Repository,get_Repository1,get_Repository2
from GAN_main import Tree_loss, trees_loss, tree_loss, train_epoch, GANLoss, GANLoss2,GANLoss3, normlize,read_file \
#,get_samples, generate_samples1, get_cluster_score
from get_subtree_structures1 import get_list_wbk

def get_bank_size(e):
    if e < 5:
        bank_size = [1, 2]
    elif e >=5 and e < 10:
        bank_size = [2, 6]
    elif e >=10 and e < 20:
        bank_size = [3, 10]
    elif e >=20 and e < 35:
        bank_size = [4, 20]
    elif e >=35 and e < 50:
        bank_size = [5, 35]
    elif e >=50 and e < 70:
        bank_size = [6, 50]
    elif e >=70 and e < 100:
        bank_size = [7, 70]
    elif e >=100 and e < 200:
        bank_size = [8, 100]
    elif e >=200 and e < 300:
        bank_size = [9, 200]
    elif e >=300:
        bank_size = [10, 300]
    return bank_size

def remove_0(l0, l=0):
        len0 = len(l0)
#        print('\nl0_len: ', len0)
#        print('l: ', l)
        if 0 in l0:
#            print('l0 with 0:   ', l0)
            l0.remove(0)
            l += len0 - len(l0)
#            print('l0 remove 0: ', l0)
#            print('l: ', l)
            l0, l = remove_0(l0, l)
            return l0, l
        else:
            return l0, l

# 写入三个文件，id，subtree，img_dir
def generate_samples3(model, batch_size, generated_num, output_file,x_info,x_ids,start_id_list,end_id_list,bank_dict):
    samples = []
    samples1 = []
    for _ in range(int(generated_num / batch_size)):
        start_st = random.sample(start_id_list, batch_size)
        start_st = np.expand_dims(start_st, axis=1) 
        start_st = Variable(torch.Tensor(start_st).long())
        sample = model.sample(BATCH_SIZE, g_sequence_len, start_st).cpu().data.numpy().tolist()
        samples.extend(sample)
    samples1,samples_tree,samples_imgdir,samples0,real_DT,samples1_e,samples_lenth = get_samples2(samples,x_info,x_ids,start_id_list,end_id_list,bank_dict)
    samples1 = samples1.cpu().data.numpy().tolist()
    
    with open(output_file+'.txt', 'w', encoding="utf-8") as fout:
        for sample in samples1:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)
    with open(output_file+'imgdir.txt', 'w', encoding="utf-8") as fout:
        for sample in samples_imgdir:
            if sample in bank_dict.values():
                string = sample
            else:
                string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)
    with open(output_file+'_no_padding.txt', 'w', encoding="utf-8") as fout:
        for sample in samples0:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)
            
    with open(output_file+'_e.txt', 'w', encoding="utf-8") as fout:
        for sample in samples1_e:
            fout.write('%s\n' % sample)
    return samples_lenth

def generate_samples4(model, batch_size, generated_num, output_file,x_info,x_ids,start_id_list,end_id_list,bank_dict,pre_st):
    samples = []
    samples1 = []
    for _ in range(int(generated_num / batch_size)):        
        start_st = [pre_st for c in range(batch_size)]
        start_st = Variable(torch.Tensor(start_st).long())
        sample = model.sample(BATCH_SIZE, g_sequence_len, start_st).cpu().data.numpy().tolist()
        samples.extend(sample)
    samples1,samples_tree,samples_imgdir,samples0,real_DT,samples1_e,samples_lenth = get_samples2(samples,x_info,x_ids,start_id_list,end_id_list,bank_dict)
    samples1 = samples1.cpu().data.numpy().tolist()
    
    with open(output_file+'.txt', 'w', encoding="utf-8") as fout:
        for sample in samples1:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)
    with open(output_file+'imgdir.txt', 'w', encoding="utf-8") as fout:
        for sample in samples_imgdir:
            if sample in bank_dict.values():
                string = sample
            else:
                string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)
    with open(output_file+'_no_padding.txt', 'w', encoding="utf-8") as fout:
        for sample in samples0:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)
            
    with open(output_file+'_e.txt', 'w', encoding="utf-8") as fout:
        for sample in samples1_e:
            fout.write('%s\n' % sample)
    return samples_lenth

def get_samples2(samples,x_info,x_ids,start_id_list,end_id_list,bank_dict):
    # 根据height累加对生成samples进行裁剪      
    # 统计一下各种情况出现的次数，改进      
    samples_height = []
    samples_lenth = []
    samples1 = []
    samples1_tree = []
    samples1_imgdir = []
    samples1_pad = []
    samples1_e = []       # 记录end信息
    s_num = 0
    end_num = 0
    exceed_num = 0
    _0num = 0             # 记录有几个sample种生成0
    real_DT = []
    for sample in samples: # samples里由batch_size个sample
        h = 0 # 总体高度
        i = 0 # 有效长度(拼接的subtree数量)
        s = []
        s_tree = ''
        img_dir = []
        s_num += 1
        _num_0 = 0 # 每个sample种有几个0
#        print('sample: ', sample)
        if not isinstance(sample,list):
            sample = sample.cpu().numpy().tolist()
        
#        if 0 in sample:
#            print('sample0    : ', sample)
#            _num_0 = len(sample) # 每个sample种有几个0
#            _0num += 1
#            sample.remove(0) # 去掉所有0
#            _num_0 = _num_0 - len(sample)
#            print('sample_del0: ', sample)
        
        sample, _num_0 = remove_0(sample)
        
        for id1 in sample: # 遍历一个sample里的subtree
            if i ==0: # 第一个不可能是bank
                real_DT.append(x_info[id1-11][0]) # 记录以该start_list的subtree开头的real_world_data的UI_id
            # (1)判断后续是否拼接到start_list的subtree
            if id1 in start_id_list and i>0: # start_list中的subtree不作为往后拼接的对象
                continue                     # 以start_id_list中st结尾立刻结束不拼接
            # (2)判断是否为bank
            if id1 <= 10:
#                print('bank: ', id1)
#                print('sam_bank: ', sample)
                h += bank_dict[str(id1)]
            else:
                h += int(x_info[id1-11][3])
            # (3)判断总高度
            if h > 2560:
                exceed_num += 1
                samples1_e.append(-1) # 超标非end结尾
                break
            i += 1 # 有效长度+1
            s.append(id1)
            if id1 <= 10:
                img_dir.append(bank_dict[str(id1)])
            else:
                s_tree += x_info[id1-11][1]
                img_dir.append(x_ids[id1-11])
            if id1 in end_id_list:  # 如果是end，则加上之后就终止(以此结尾)
                _ui = os.path.basename(x_ids[id1-11]).split('_')[0]
                end_num += 1
                samples1_e.append(_ui) # 正常以end结尾，记录ui号
                break
            
        samples_height.append(h)
        samples_lenth.append(i)   # 有效长度
        samples1.append(s)        # no padding的list
        p = len(sample)-i+_num_0  # 需要padding的长度
        # 加padding部分
        samples1_pad.append(np.pad(s, (0,p)))
        samples1_tree.append(s_tree)
        samples1_imgdir.append(img_dir)
    
#    print('\ns_num: ', s_num)            # 所有数量
#    print('end_num: ', end_num)          # 以end正常结尾
#    print('exceed_num: ', exceed_num)    # 超出高度限制的数量
#    print('_0num: ', _0num)              # 出现0的数量
    
    samples = torch.LongTensor(samples1_pad)
    return samples,samples1_tree,samples1_imgdir,samples1, real_DT,samples1_e,samples_lenth

def get_cluster_score(x_e, x_e_label):
    n = len(x_e_label) # 标记class
#    print('x_e_labels:', n)
    v = []
    j = 0
    for m in range(n): # n个样本
        z = 0
        per_num = len(x_e_label[m])
        label_num = [os.path.basename(os.path.dirname(x)) for x in x_e_label[m]] # 有多少label
        n_class = []
        label_num1 = []
        i = -1
#        print('label_num', len(label_num),label_num)
        for x in label_num:
            if x not in n_class:
                n_class.append(x)
                i += 1
            color = np.ones((1,), dtype=int) * i
            if label_num1 == []:
                label_num1 = color
            else:
                label_num1 = np.concatenate((label_num1, color), axis = 0)
        # output label_num1
        
        # kmeans 聚类
#        print('per_num: ', per_num)
        if per_num ==1: # 只有1个subtree不合格，返回最大loss
#            print(j, 'per_num ==1, add 0')
            v.append(1.0)
        else:            
#            if len(n_class)>1:
#                print('n_class: ', n_class)
#                print('n_class(k_clusters)', len(n_class))
            k_clusters =  len(n_class)
            if(k_clusters >1):
                km = KMeans(n_clusters=k_clusters, random_state=0).fit(x_e[m])
                _clusters = km.labels_            
                #将每个学习到的簇标签和真实标签进行匹配 
                labels = np.zeros_like(_clusters)
                for i in range(k_clusters):
                    mask = (_clusters == i)
                    labels[mask] = mode(label_num1[mask])[0]
                _v = float(metrics.homogeneity_score(label_num1, labels))
#                v.append(-np.log(float(1 - metrics.homogeneity_score(label_num1, labels)))) # output Homogeneity， 还可以用其他的metrics
#                v.append(float(1 - metrics.homogeneity_score(label_num1, labels))) # output Homogeneity， 还可以用其他的metrics
                v.append(np.exp(-_v))
#                print(j, 'per_num >1, add ', str(np.exp(-_v)))
            else:
                z += 1
                v.append(0.0) # 多个subtree但属于一个class，返回最小loss
#                print(j, 'per_num >1 and k_clusters==1')
##            print(j, 'per_num >1 and k_clusters==1: ', z)
        j += 1
    val = np.mean(v)
    return(val)



# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
#parser.add_argument('--cuda', action='store', default=None, type=int)
parser.add_argument('--cuda', action='store', default=0, type=int)
opt = parser.parse_args()
print(opt)

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 32
#BATCH_SIZE = 4
#TOTAL_BATCH = 100
TOTAL_BATCH = 10 # 迭代次数
#TOTAL_BATCH = 500

POSITIVE_FILE = 'real' # real_data是由oracle model产生的real-world数据
EVAL_FILE = 'eval' # eval_file是由Generator产生，用来评价Generator和oracle model相似性所产生的数据
#VOCAB_SIZE = 5000 # 词向量个数，这里应该是template或者subtree个数 train_embedding=(34964,256), 实际107319 subtrees
#PRE_EPOCH_NUM = 120
PRE_EPOCH_NUM = 50

if opt.cuda is not None and opt.cuda >= 0:
    print('opt.cuda is not None')
#    torch.cuda.set_device(0)
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True
else:
    print('opt.cuda is None')


# Genrator Parameters
g_emb_dim = 32      # Embedding 维度
g_hidden_dim = 32   # nn.LSTM(emb_dim, hidden_dim)
#g_sequence_len = 10 # 长度为20的句子
#g_sequence_len = 15 
g_sequence_len = 30

# Discriminator Parameters
d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
#d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
#d_num_filters = [100, 200, 200, 100, 100, 100, 160, 160]
d_dropout = 0.5
d_num_class = 2 # D二分类

if __name__ == '__main__':       
    
#     Adversarial Training
    random.seed(SEED)
    
    file_csv = r'F:\2017\zhu\RicoDataset\app_details.csv'    
    cd_img = r'D:\zhu\chen1\data\pick8\p_app_resize_Td_sts_c_resize512_noui1'    
    txt_img = r'D:\zhu\chen1\data\pick8\aTrees_dict_app'
    st_dir = r'D:\zhu\chen1\data\pick8\p_app_resize_Td_sts'
    db_dir = r'D:\zhu\chen1\data\pick8\st_bank_app'    

#    m_save_path = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\model' # 3 loss
#    NEGATIVE_FILE = '.\samples' # 是由Generator产生,用作pre-train G
    
#    m_save_path = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\model1' # 2 loss e_loss
#    NEGATIVE_FILE = '.\samples1'
    
    m_save_path = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\model2' # 2 loss t_loss
    NEGATIVE_FILE = '.\samples2' 
    
    '''按categories获取apps'''
    appsl, appsd = get_s_app(file_csv, st_dir)
    
    appsl1 = []
    for (k,v) in appsd.items():
        appsl1.append([k,len(v)])
    appsl1 = sorted(appsl1, key=lambda x: x[1], reverse=True) # 按照每个cat中app数量由高到低排序
    
    _ns = []
#    _n = 0; _ns.append(_n) # News & Magazines
#    _n = 1; _ns.append(_n) # Books & Reference
#    _n = 2; _ns.append(_n) # Shopping
#    _n = 3; _ns.append(_n) # Communication 
#    _n = 4; _ns.append(_n) # Entertainment
    _n = 5; _ns.append(_n) # Travel & Local
#    _n = 6; _ns.append(_n) # Lifestyle
#    _n = 7; _ns.append(_n) # Sports
#    _n = 8; _ns.append(_n) # Social
#    _n = 9; _ns.append(_n) # Music & Audio
#    _n = 12; _ns.append(_n) # Weather
    
    mul_loss = True
#    mul_loss = False
    
    c_apps = []
    c_cats = []
    for _n in _ns:
        c_cat = appsl1[_n][0]
        c_cats.append(c_cat) # 选择第_n个cat
        c_apps += appsd[c_cat]
    print('\n', c_cats)
    
    # 读取db_dir中带bank的ui_sts
    real_data_bk = get_list_wbk(db_dir)
    real_data_bk_c = [c for c in real_data_bk if c[0] in c_apps] # 根据category筛
    
    emb_file = r'D:\SpyderWorkSpace\GUI_Ganerator\data\categories_app_emb'
    '''load embs'''
    starttime = time.time()
    x_ids = []
    x_emb = []
    c_cat = ''
    for _cat in c_cats:        
        c_cat_emb_file = os.path.join(emb_file, str(_cat)+'.txt')
        _x_ids,_x_emb = read_data(c_cat_emb_file) # loading: 851.8156 s
        x_ids += _x_ids
        if x_emb == []:
            x_emb = _x_emb
            c_cat = _cat
        else:
            x_emb = np.concatenate((x_emb, _x_emb), axis = 0)
            c_cat = c_cat + '_and_' + _cat
    # x_ids与x_emb要在一起
    endtime = time.time(); dtime = endtime - starttime    
    print("\nTime for loading training embedding：%.8s s" % dtime)
    ''''''
    
    # sample 的folder
    NEGATIVE_FILE = os.path.join(NEGATIVE_FILE, c_cat)
    print('NEGATIVE_FILE: ', NEGATIVE_FILE)
    if not os.path.exists(NEGATIVE_FILE):
        os.mkdir(NEGATIVE_FILE)
    
    train_uis = []
    for app in os.listdir(cd_img): # (1) subtree img dir
        if app in c_apps:
            app_dir = os.path.join(cd_img,app)
            for ui in os.listdir(app_dir):
                ui_dir = os.path.join(app_dir,ui)
                train_uis.append(ui_dir)
    
    '''读取dom_tree'''    
    train_uis1 = []
    for app1 in os.listdir(txt_img): # (2) subtree txt dir
        if app1 in c_apps:
            app_dir1 = os.path.join(txt_img,app1)
            for ui1 in os.listdir(app_dir1):
                ui_dir1 = os.path.join(app_dir1,ui1)
                train_uis1.append(ui_dir1)
                
    train_uis_tree,train_templates_list1,train_templates_dict1 = get_Repository2(train_uis1)
    train_DT = []
    train_DT1 = []
    train_uis_tree = sorted(train_uis_tree,key=lambda x: x[0].split('.txt')[0].split('_')[-1]) # 统一按ui号排序
    for ui in train_uis_tree:
        s = ''
        if len(ui[1]) == 1:
            continue        
        for (k,v) in ui[1].items():
            s += v
        train_DT.append(s)
        # 需要按ui[1]的key排序
        s1 = ''
        s2 = []
        ui_sorted = sorted(ui[1].items(),key=lambda x: (len(x[0]), x[0])) # 统一排序规则，先长度，后标号
        for u in ui_sorted:
            s1 += u[1]
            s2.append(u[0])
        train_DT1.append([s1,s2])
        ui_id = ui[0].split('_')[-1].split('.txt')[0]
    train_DT0 = train_DT
    train_DT = [x[0] for x in train_DT1]
    
    x_info = get_ui_info(train_uis,txt_img)
    x_info_ids = [x[4] for x in x_info]           
    train_DT_id = [x[0].split('_')[-1].split('.txt')[0] for x in train_uis_tree]
            
    start_id_list = [] # 
    end_id_list = []
    
    # load real_data_id in each ui_st_txt 获取真实数据
    starttime = time.time()
    real_data_id = []
    real_data = []

    x_ids2 = [[x[0], x[4], st_dir+'\\'+os.path.basename(os.path.dirname(x[4]))+'\\'+os.path.basename(x[4]).split('_')[0]+'\\'+os.path.basename(x[4]).split(os.path.basename(x[4]).split('_')[0]+'_')[-1],x[5]] for x in x_info]
    uis = [x[0].split('_')[0] for x in x_ids2]
    uis = list(set(uis))
    uis = sorted(uis,key=lambda x: x) # 统一按ui号排序
    fit_ui_banks = []
    for ui in uis:
        fit_uis = [x for x in x_ids2 if ui == x[0].split('_')[0]]
#        fit_uis = sorted(fit_uis,key=lambda x: (int(x[3].split(',')[1]), len(x[0]))) # 排序之后
        fit_uis = sorted(fit_uis,key=lambda x: (int(x[3].split(',')[1]), 1/(int(x[3].split(',')[3])-int(x[3].split(',')[1]))))
        fit_uis_bk = [x for x in real_data_bk_c if ui == x[1]][0][-1]
        fit_ui_banks.append(fit_uis_bk)
        st_id_list = []
        st_list = []
        for u in fit_uis:      
            '''新bank'''
            #----------------------------------------------#
            st_id_list.append(x_ids2.index(u)+11)
            st_list.append(u)            
            if u == fit_uis[-1]:
                continue            
            st_id = u[0][len(u[0].split('_')[0])+1:] # 获取st_id去fit_uis_bk里找            
            _id = fit_uis_bk.index(st_id)            # 获取index
#            print('st_id: ', _id, ',', st_id)                         
            for i in range(len(fit_uis_bk)):
#                if _id+i+1 >= len(fit_uis_bk):       # 去掉最后一个
#                    break
                st = fit_uis_bk[_id+i+1]
                if st.split('_')[0] == 'bk':
#                    print('bk: ',st)
                    bk = get_bank_size(int(st.split('_')[1]))
#                    print(bk)
                    st_id_list.append(bk[0])         # 可能一个st后接不止一个bank(由于中间的st可能之前被删除了)
                    st_list.append(bk)
                else:
                    break
            #----------------------------------------------#
            
        if len(st_list) == 1: # 如果只有一个subtree的real world data 就不要了 test:1286 - 1136
            continue
        real_data_id.append(st_id_list)
        real_data.append(st_list)
        start_id_list.append(st_id_list[0])
        end_id_list.append(st_id_list[-1])
    
    real_data_id0 = real_data_id.copy()
    real_data_id = [x[:g_sequence_len] for x in real_data_id] # 长度截取
    real_data_id1 = [np.pad(x, (0,g_sequence_len - len(x))) for x in real_data_id]
    endtime = time.time(); dtime = endtime - starttime    
    print("\nTime for loading real world data：%.8s s" % dtime)           
    
#    bank_ls.sort()
#    bank_lsd = sorted(bank_lsd.items(), key = lambda x:x[1])
    
#    bank_dict = {'1':2, '2':6, '3':10, '4':16, '5':22, '6':32, '7':43, '8':53, '9':65, '10':100}
    bank_dict = {'1':2, '2':6, '3':10, '4':20, '5':35, '6':50, '7':70, '8':100, '9':200, '10':300}

    GENERATED_NUM = len(real_data_id1) # 
    print('\nGENERATED_NUM,real_data_id1', GENERATED_NUM)
        
    VOCAB_SIZE = len(x_info_ids)+1+10 # 加padding的0， 和10种size的bank
    print('\nVOCAB_SIZE:',VOCAB_SIZE)
    print('real_vocab_size: ', len(x_info_ids))
    
    # 根据x_info_ids更新x_ids和x_emb
    starttime = time.time()
    x_index = []
    for i in range(len(x_ids)):
        if x_ids[i] not in x_info_ids:
            x_index.append(i)
    
    x_ids_not_in_info = [x_ids[i] for i in x_index]
#    x_ids = [m for m in x_ids if m not in x_ids_not_in_info]
    x_ids = [x_ids[i] for i in range(len(x_ids)) if i not in x_index]
    
    x_emb_not_in_info = [x_emb[i] for i in x_index]
#    x_emb = [m for m in x_emb if m not in x_emb_not_in_info]
    x_emb = [x_emb[i] for i in range(len(x_emb)) if i not in x_index]
    endtime = time.time(); dtime = endtime - starttime    
    print("\nTime for unifying x_info and x_emb：%.8s s" % dtime)
    print('\n')   
           
    reduced_data1 = PCA(n_components=2).fit_transform(x_emb)
#    reduced_data1 = normlize(reduced_data1)

    '''
    构建网络模型
    '''
    #与原来相比，没有oracle,即target_lstm，lstm循环次数受限制于data的height累加是否超过额定值    
    #Define Networks
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)    
    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    
    for name, param in generator.named_parameters():
        print('name: ',name, 'param: ',param.shape)
    
    rollout = Rollout(generator, 0.8)
    print('\n#####################################################')
    print('Start Adeversatial Training......')
    if mul_loss:
        gen_gan_loss = GANLoss2()
#        gen_gan_loss = GANLoss3()        
    else:
        gen_gan_loss = GANLoss() # 最小化policy gradient损失
        
    if opt.cuda:
        gen_gan_loss = gen_gan_loss.cuda()
        
    gen_gan_optm = optim.Adam([{"params":generator.parameters()},
                               {"params":gen_gan_loss.parameters()}],lr=0.05)
#    gen_gan_optm = optim.Adam([{"params":generator.parameters()},
#                               {"params":gen_gan_loss.parameters()}],lr=0.01)
    
    dis_criterion = nn.NLLLoss(reduction='sum')# negative log likelihood loss
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    
    # 新建cat_folder
    log_g_file = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\log_cvs'
    log_g_file = os.path.join(log_g_file, c_cat)
    if not os.path.exists(log_g_file):
        os.mkdir(log_g_file)
    log_g_file = os.path.join(log_g_file, 'loss_g.csv')
    
    log_d_file = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\log_cvs'
    log_d_file = os.path.join(log_d_file, c_cat)
    if not os.path.exists(log_d_file):
        os.mkdir(log_d_file)
    log_d_file = os.path.join(log_d_file, 'loss_d.csv')
    
    headers_g = ['epoch','loss1', 'emb_loss', 'tree_loss','mul_loss']
    headers_d = ['epoch','d_loss','f_d_loss']
    rows_g = []; rows_d = []
    with open(log_g_file,'w') as f1:
        f_csv1 = csv.writer(f1)        
        f_csv1.writerow(headers_g)
        with open(log_d_file,'w') as f2:
            f_csv2 = csv.writer(f2)
            f_csv2.writerow(headers_d)
            
            for total_batch in range(TOTAL_BATCH):
                print('\nepoch=============: ', total_batch)
                starttime = time.time()
                if not os.path.exists(os.path.join(NEGATIVE_FILE,str(total_batch))):
                    os.mkdir(os.path.join(NEGATIVE_FILE,str(total_batch)))
                    
                ## Train the generator for one step 训练G
                for it in range(1): # 训练1此G，训练4次D
#                    samples = generator.sample(BATCH_SIZE, g_sequence_len) # 这应该是由G产生的fake data，输入应该是读取数据                    
                    start_st = random.sample(start_id_list, BATCH_SIZE)
                    start_st = np.expand_dims(start_st, axis=1) 
                    start_st = Variable(torch.Tensor(start_st).long())
#                    x = Variable(torch.zeros((BATCH_SIZE, 1)).long())
#                    print('start_st: ', start_st)
#                    print('start_zero: ', x)
                    samples = generator.sample(BATCH_SIZE, g_sequence_len, start_st)
                    samples1,samples_tree,samples_imgdir,samples0,real_DT,samples1_e,samples_lenth = get_samples2(samples,x_info,x_ids,start_id_list,end_id_list,bank_dict)
                    
                    if generator.use_cuda:
                        samples = samples1.cuda()
                        
                    if mul_loss:
                        #------------加入emb_loss--------------------#
#                        # 生成样本的embs
#                        x_embs1 = []
#                        x_info_ids1 = []
#                        for samp in samples0:
#                            embs1 = []
#                            info_ids1 = []
#                            for n in samp:
#                                if n > 10:
#                                    embs1.append(reduced_data1[n-11])
#                                    info_ids1.append(x_info_ids[n-11])
#                            x_embs1.append(embs1)
#                            x_info_ids1.append(info_ids1)
#                            
#                        c_loss = get_cluster_score(x_embs1, x_info_ids1)
#                        c_loss = torch.Tensor([c_loss])
#                        c_loss.requires_grad_()
#                        c_loss = c_loss.cuda()
#                        print('c_loss:', c_loss)
                        
                        #------------加入tree_loss--------------------#
                        s_len = len(samples_tree)                    
    #                    real_DT, train_DT_id, train_DT
                        _loss = []
                        _losses = 0
                        for i in range(s_len): # 32
                            _uid = real_DT[i].split('_')[0]
                            tt1_i = train_DT_id.index(_uid)
                            tt1 = train_DT[tt1_i]
                            tree1 = Tree.from_text(tt1)
                            tt2 = samples_tree[i]
                            tree2 = Tree.from_text(tt2)
                            _apted = APTED(tree1, tree2, Config())
                            ted = _apted.compute_edit_distance()
                            _loss.append(ted)
                            _losses += ted
    #                        print('ted: ', ted)
                        
    #                    t_loss = torch.Tensor([_losses])
                        t_loss = torch.mean(torch.Tensor([_loss]))
                        t_loss.requires_grad_()
                        t_loss = t_loss.cuda()
                        print('t_loss:', float(t_loss))
                        print('total_t_loss: ', _losses)
                        #-------------------------------------------#   
                    
                    # construct the input to the genrator, add zeros before samples and delete the last column
                    zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
                    if samples.is_cuda:
                        zeros = zeros.cuda()
                    
                    inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
#                    print('inputs.shape', inputs.shape)
                    
                    targets = Variable(samples.data).contiguous().view((-1,))
                    # calculate the reward
                    rewards = rollout.get_reward(samples, 16, discriminator)
#                    print('rewards: ', rewards.shape)
                    
                    rewards = Variable(torch.Tensor(rewards))
                    rewards = torch.exp(rewards).contiguous().view((-1,))
#                    print('rewards exp: ', rewards.shape)
                    if opt.cuda:
                        rewards = rewards.cuda()
                    prob = generator.forward(inputs)
                    
                    if mul_loss:
#                        loss, loss1 = gen_gan_loss(prob, targets, rewards, c_loss, t_loss) # 3个loss
#                        loss, loss1 = gen_gan_loss(prob, targets, rewards, c_loss) # 3个loss
                        loss, loss1 = gen_gan_loss(prob, targets, rewards, t_loss) # 3个loss
                    else:
                        loss = gen_gan_loss(prob, targets, rewards) # 经过policy gradient计算后的loss
#                    print('g_loss', loss)
                                                        
                    gen_gan_optm.zero_grad() # 将module中的所有模型参数的梯度设置为0.
                    loss.backward()
                    gen_gan_optm.step()
#                    print('one_epoch_g_loss ', loss.item())             
                    
                    # 写log_cv
#                    if mul_loss:
#                        f_csv1.writerows([[total_batch,loss1.item(),c_loss.item(), t_loss.item(),loss.item()]])
#                    else:
#                        f_csv1.writerows([[total_batch,loss.item()]])
                
                rollout.update_params() # 循环一次后更新，rollout把传入的generator的参数以0.8的更新rate进行update
                # 新建cat_folder
#                g_save_path = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\model\generator'+str(total_batch)+'.pkl'

                g_save_path = os.path.join(m_save_path, c_cat)
                if not os.path.exists(g_save_path):
                    os.mkdir(g_save_path)
                g_save_path = os.path.join(g_save_path, 'generator'+str(total_batch)+'.pkl')
                torch.save(generator.state_dict(), g_save_path)
                
                # 训练D
                print('\n')
                for p in range(4):
#                for p in range(1):
                    # G生成fake data，对比所有train_uis的Dom_tree
                    NEGATIVE_FILE1 =  NEGATIVE_FILE + '\\' + str(total_batch) + '\\gene'
                    samples_lenth = generate_samples3(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE1,x_info,x_ids,start_id_list,end_id_list,bank_dict)
                                        
                    NEGATIVE_FILEtxt = NEGATIVE_FILE + '\\' + str(total_batch) + '\\gene.txt'                    
                    
                    dis_data_iter = DisDataIter(real_data_id1, NEGATIVE_FILEtxt, BATCH_SIZE)# 所有真实样本
                    for q in range(2):
#                    for q in range(1):
                        
                        total_loss = 0.
                        total_words = 0.
                        
                        n = 0 # 70
                        for (data, target) in dis_data_iter:
                            n+=1
                            data = Variable(data)
                            target = Variable(target)
                            if opt.cuda:
                                data, target = data.cuda(), target.cuda()
                            target = target.contiguous().view(-1) # 0或1标签
                            pred = discriminator.forward(data) # 生成样本为real-world数据的概率
                            # (1) Loss1
                            loss = dis_criterion(pred, target) # negative log likelihood loss                            
                            total_loss += loss.item()
                            total_words += data.size(0) * data.size(1) # 539520        
                            
                            dis_optimizer.zero_grad() # 梯度初始化为零
                            loss.backward()       # 反向传播求梯度
                            dis_optimizer.step()      # 更新所有参数            
                        
                        dis_data_iter.reset() # 一次epoch完成
                        f_loss = math.exp(total_loss/ total_words) # 超过709 会报错
                        # 新建cat_folder
#                        d_save_path = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\model\discriminator'+str(total_batch)+'.pkl'                        
                        
                        d_save_path = os.path.join(m_save_path, c_cat)
                        if not os.path.exists(d_save_path):
                            os.mkdir(d_save_path)
                        d_save_path = os.path.join(d_save_path, 'discriminator'+str(total_batch)+'.pkl')                        
                        torch.save(discriminator.state_dict(), d_save_path)
                        f_csv2.writerows([[str(total_batch)+'_'+str(p)+'_'+str(q),str(total_loss),str(f_loss)]])
#                print('total_d_loss ', total_loss)
#                print('f_d_loss ',f_loss)
        
                # G和D都训练完了
                endtime = time.time()
                dtime = endtime - starttime
                print("\nTime for one adeversatial training epoch：%.8s s" % dtime)
                     
    
     
    
    