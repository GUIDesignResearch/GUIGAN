# -*- coding: utf-8 -*-

import os,time
import random
import math
import numpy as np
import torch

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
        if 0 in l0:
            l0.remove(0)
            l += len0 - len(l0)
            l0, l = remove_0(l0, l)
            return l0, l
        else:
            return l0, l

def get_samples(samples,x_info,x_ids,start_id_list,end_id_list,bank_dict):  
    samples_height = []
    samples_lenth = []
    samples1 = []
    samples1_tree = []
    samples1_imgdir = []
    samples1_pad = []
    samples1_e = []  # end info
    s_num = 0
    end_num = 0
    exceed_num = 0
    _0num = 0             
    real_DT = []
    for sample in samples: 
        h = 0 # total height
        i = 0 # len(subtree num)
        s = []
        s_tree = ''
        img_dir = []
        s_num += 1
        _num_0 = 0 # 0 num
        if not isinstance(sample,list):
            sample = sample.cpu().numpy().tolist()        
        sample, _num_0 = remove_0(sample)
        
        for id1 in sample: #
            if i ==0: 
                real_DT.append(x_info[id1-11][0]) # 记录以该start_list的subtree开头的real_world_data的UI_id
            # (1)判断后续是否拼接到start_list的subtree
            if id1 in start_id_list and i>0: # start_list中的subtree不作为往后拼接的对象
                continue                     # 以start_id_list中st结尾立刻结束不拼接

            if id1 <= 10:
                h += bank_dict[str(id1)]
            else:
                h += int(x_info[id1-11][3])
            if h > 2560:
                exceed_num += 1
                samples1_e.append(-1) 
                break
            i += 1 
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
        # padding
        samples1_pad.append(np.pad(s, (0,p)))
        samples1_tree.append(s_tree)
        samples1_imgdir.append(img_dir)
    
    samples = torch.LongTensor(samples1_pad)
    return samples,samples1_tree,samples1_imgdir,samples1, real_DT,samples1_e,samples_lenth

def read_trees(ui_file): 
    with open(ui_file, 'r', encoding='gbk') as f:        
        f_content = f.read() 
        aTrees2 = f_content.split(',\n')
        aTrees2.remove(aTrees2[-1])
        aTrees2_dict = {}
        for t in aTrees2:
            k1 = t.split(':')[0].strip()
            v1 = t.split(':')[1].strip()
            aTrees2_dict[k1] = v1
        
        dict_temp = f_content.split(';\n')[1]
        fc_dict = {}
        for line in dict_temp.split('\n')[:-1]:            
            line = line.strip()
            k = line.split(' [')[0]
            v = line.split(' [')[1][:-1]
            values = v.split(',')
            fc_dict[k] = [value.strip()[1:-1] for value in values]
    
    return aTrees2_dict,fc_dict

def get_template_T(ui_n,ui_tree):
    subtrees_count = 0
    templates_list = []
    templates_dict = {}
    for (k,v) in ui_tree.items():
        subtrees_count+=1
        if v not in templates_list:
            templates_list.append(v)
            templates_dict[v] = [ui_n+','+k]
        else:
            templates_dict[v].append(ui_n+','+k)    
    return subtrees_count,templates_list,templates_dict

def get_ui_name(cd):
    a2 = os.path.basename(cd)
    a3 = cd.split(a2)[0][:-1]
    a4 = os.path.basename(a3)
    return a4+'_'+a2

def get_Repository_T(ui):
    train_uis_tree = []
    ui_tree,ui_dict = read_trees(ui) 
    ui_n = get_ui_name(ui)
    train_uis_tree.append([ui_n,ui_tree,ui_dict])        
    train_subtrees_count,train_templates_list,train_templates_dict=get_template_T(ui_n,ui_tree)
    return train_uis_tree,train_subtrees_count,train_templates_list,train_templates_dict


def get_Repository(train_uis):
    starttime = time.time()
    train_uis_tree = []
    train_subtrees_count = 0
    train_templates_list = []
    train_templates_dict = {}
    
    for ui in train_uis:
        ui_t, st_c,sttl,sttd = get_Repository_T(ui)
        train_uis_tree+=ui_t
        train_subtrees_count+=st_c
        train_templates_list+=sttl
        train_templates_dict.update(sttd)
    
    print('\nsubtrees_count:',train_subtrees_count)
    print('train_templates_count:', len(train_templates_list))
    endtime = time.time()
    dtime = endtime - starttime    
    print("\nTime for getting Repository：%.8s s" % dtime)
    return train_uis_tree,train_templates_list,train_templates_dict

def get_list_wbk(db_dir):
    real_data_bank = []
    for app in os.listdir(db_dir):
        app_dir = os.path.join(db_dir, app)
        for ui in os.listdir(app_dir):
            ui_file = os.path.join(app_dir, ui)
            with open(ui_file, 'r', encoding="utf-8") as f:
                lines = f.readlines()
            lis = []
            for line in lines:
                l = line.split(',')[0].strip()
                lis.append(l)
            real_data_bank.append([app, ui.split('.txt')[0],lis])
    return real_data_bank


