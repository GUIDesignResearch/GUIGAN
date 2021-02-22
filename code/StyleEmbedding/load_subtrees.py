# -*- coding: utf-8 -*-
"""
randomly select subtree pairs to train the SiameseNet
"""

import os,random
import numpy as np
from itertools import combinations
from load_data import load_dataset, write_file,read_valid_file

def write_pair(_pair_file, train_intra_pair, train_inter_pair):
    for train_pair in train_intra_pair:
        info = train_pair[0]+','+train_pair[1]+ '\n'
        write_file(_pair_file, info)
    write_file(_pair_file, ';')
    print(';')
    for train_pair in train_inter_pair:
        info = train_pair[0]+','+train_pair[1]+ '\n'
        write_file(_pair_file, info)

def read_pair(_pair_file):
    train_intra_pair, train_inter_pair = [], []
    intra = True
    with open(_pair_file, 'r') as f:
        for line in f.readlines():
            if(line is '\n'):
                continue
            lineseparate = line.split(';')
            if(len(lineseparate)>1):
                intra = False
                line = lineseparate[1]
            linelist = line.split(',')
            if(intra): 
                train_intra_pair.append([linelist[0], linelist[1].strip()])
            else:                
                train_inter_pair.append([linelist[0], linelist[1].strip()])
    return train_intra_pair, train_inter_pair

def get_soft_all_image_pair(dictionary, available_apps,ui_c=20,s_num=10): 
    _count = 0
    intra_pair = []
    available_apps1 = []
    print('available_apps_nums: ', len(available_apps))
    for i in range(len(available_apps)):
        current_app = available_apps[i]
        current_app_dirs = dictionary[current_app]
        arr = np.arange(len(current_app_dirs))
        np.random.shuffle(arr)
        if len(arr)>=ui_c:
            available_apps1.append(available_apps[i])
            _count+=1
            arr_c = arr[:ui_c] # Randomly select 10 subtrees in each app
            
            m = int(len(arr_c)/2)
            for j in range(m):
                intra_pair.append([current_app_dirs[arr_c[j]],current_app_dirs[arr_c[j+m]]])
        else:
            continue
    print('enable app_num:',_count,'containing',len(intra_pair),'intra_pairs')
    
    
    inter_pair = []
    inter = np.arange(len(available_apps1))
    np.random.shuffle(inter)
    print('\ninter: ', inter)    
    
    mi = int(len(inter)/2)
    for m in range(mi):
        app1 = available_apps1[inter[m]]
        app2 = available_apps1[inter[m+mi]]
        
        app1_dirs = dictionary[app1]
        app2_dirs = dictionary[app2]        
        ar1 = random.sample(range(0, len(app1_dirs)-1), s_num)
        ar2 = random.sample(range(0, len(app2_dirs)-1), s_num)
        inter_pair.append([app1_dirs[ar1[0]], app2_dirs[ar2[0]]])
        [inter_pair.append([app1_dirs[ar1[i]], app2_dirs[ar2[i]]]) for i in range(s_num)]
        
    for m in range(mi):
        app1 = available_apps1[inter[m]]
        app2 = available_apps1[inter[2*mi-1-m]]
        
        app1_dirs = dictionary[app1]
        app2_dirs = dictionary[app2]        
        ar1 = random.sample(range(0, len(app1_dirs)-1), s_num)
        ar2 = random.sample(range(0, len(app2_dirs)-1), s_num)
        inter_pair.append([app1_dirs[ar1[0]], app2_dirs[ar2[0]]])
        [inter_pair.append([app1_dirs[ar1[i]], app2_dirs[ar2[i]]]) for i in range(s_num)]
        
    return intra_pair, inter_pair


if __name__ == "__main__":
    cd = r'.\p_app_resize_Td_sts'
    
    dictionary = load_dataset(cd)
    _file = r'.\data\data.txt'
    
    train_apps, valid_apps, test_apps = read_valid_file(_file)
    
#    train_intra_pair, train_inter_pair = get_random_image_pairs(ui_dictionary, train_apps)
    
    train_intra_pair, train_inter_pair = get_soft_all_image_pair(dictionary, train_apps)
    valid_intra_pair, valid_inter_pair = get_soft_all_image_pair(dictionary, valid_apps)
    test_intra_pair, test_inter_pair = get_soft_all_image_pair(dictionary, test_apps)

    _train_pair_file = r'.\data\train_st_pair.txt'
    _valid_pair_file = r'.\data\valid_st_pair.txt'
    _test_pair_file = r'.\data\test_st_pair.txt'
#    # write the pair data to file    
    write_pair(_train_pair_file, train_intra_pair, train_inter_pair)
    write_pair(_valid_pair_file, valid_intra_pair, valid_inter_pair)
    write_pair(_test_pair_file, test_intra_pair, test_inter_pair)

#    # read the pair data from file
    train_intra_pair1, train_inter_pair1 = read_pair(_train_pair_file)
    valid_intra_pair1, valid_inter_pair1 = read_pair(_valid_pair_file)
    test_intra_pair1, test_inter_pair1 = read_pair(_test_pair_file)
#    
    
    
    
    
    
    
    
    
    
    