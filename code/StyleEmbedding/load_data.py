# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 09:31:41 2019
(6)读取数据,分三个：train, valid, test
@author: ztm
"""

import os,random,time,csv
import numpy as np
from apted import APTED, Config
from  apted.helpers import Tree

def get_ui_name(cd):
    a2 = os.path.basename(cd)
    a3 = cd.split(a2)[0][:-1]
    a4 = os.path.basename(a3)
    return a4+'_'+a2

def load_dataset(ui_path):
    count = 0
    ui_dictionary = {}
    for app in os.listdir(ui_path):
        app_path = os.path.join(ui_path, app)
        current_app = []
        for character in os.listdir(app_path):
            character_path = os.path.join(app_path, character)
            current_app.append(character_path)
            count += 1
        ui_dictionary[app] = current_app
    print('Total_GUI_num: ', count)
    return ui_dictionary

def split_valid_datasets(data, r1 = 0.9, r2 = 0.94):
    available_apps = list(data.keys())
    number_of_apps = len(available_apps)
    train_valid_apps = []
    train_apps = []
    validation_apps = []
    test_apps = []
    print('number_of_apps', number_of_apps)
    train_indexes = random.sample(range(0, number_of_apps - 1), int(r1 * number_of_apps))
    train_indexes.sort(reverse=True)
    for index in train_indexes:
        train_valid_apps.append(available_apps[index])
        available_apps.pop(index)
    test_apps = available_apps
    
    number = len(train_valid_apps)
    validation_index = random.sample(range(0, number - 1), int(r2 * number))
    validation_index.sort(reverse=True)
    for index in validation_index:
        train_apps.append(train_valid_apps[index])
        train_valid_apps.pop(index)
    validation_apps = train_valid_apps    
    return train_apps, validation_apps, test_apps

def write_file(path_file_name,_str): # 将随机分好的train和test的apps存入txt文件
    if not os.path.exists(path_file_name):
        with open(path_file_name, "w") as f:
            print(f)
    with open(path_file_name, "a") as f:
        f.write(_str)
        
def read_valid_file(_file): # 读取随机分好的train和test的apps
    with open(_file, 'r') as f:
        apps = f.read()
    train_apps, valid_apps, test_apps = apps.split(';')[0],apps.split(';')[1],apps.split(';')[2]
    train_apps = train_apps.split(','); train_apps.pop(-1)
    valid_apps = valid_apps.split(','); valid_apps.pop(-1)
    test_apps = test_apps.split(','); test_apps.pop(-1)
    return train_apps, valid_apps, test_apps

def get_random_ui(dictionary,available_apps,n_min=1,n_max=5):
    selected_uis = []
    print('available_apps_nums: ', len(available_apps))
    for i in range(len(available_apps)):
        current_app = available_apps[i]
        current_app_dirs = dictionary[current_app]
        print('\ncurrent_app: ', current_app)
        print('available_uis_nums: ', len(current_app_dirs))
        if len(current_app_dirs) <n_min:
            continue
        arr = np.arange(len(current_app_dirs))
        np.random.shuffle(arr)
        print('random select: ', arr) # 随机取类内的一对图像
        arr = arr[:n_max]
        print('random select cut: ', arr)
        for j in range(int(len(arr))):
            selected_uis.append(current_app_dirs[arr[j]])
            print('selected ui:', current_app_dirs[arr[j]])
    return selected_uis
  
def read_file(ui_file): # 读取UI的pair，分号；用来分开intra和inter
    _uis = []
    with open(ui_file, 'r') as f:
        for line in f.readlines():
            linelist = line.split(',')
            _uis.append(linelist)
    return _uis[0][:-1]

def read_trees(ui_file): # 读取UI的pair，分号；用来分开intra和inter
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

def get_template(ui_n,ui_tree,subtrees_count,templates_list,templates_dict):
    # 统计模板重复
#    for subtree in ui_tree:
#        subtrees_count+=1
#        a_tree= subtree.split(':')[1] # v
#        if a_tree not in templates_list:
#            templates_list.append(a_tree)
#            templates_dict[a_tree] = [ui_n+','+subtree.split(':')[0]] #k
#        else:
#            templates_dict[a_tree].append(ui_n+','+subtree.split(':')[0]) #k
            
    for (k,v) in ui_tree.items():
        subtrees_count+=1
        if v not in templates_list:
            templates_list.append(v)
            templates_dict[v] = [ui_n+','+k]
        else:
            templates_dict[v].append(ui_n+','+k)    
    return subtrees_count,templates_list,templates_dict

def get_s_app(file_csv, cd, app=True):
#    file_csv = r'F:\2017\zhu\RicoDataset\app_details.csv'
    categories = []
    cv_file = csv.reader(open(file_csv,'r', encoding='UTF-8'))
    #print(cv_file)
    for stu in cv_file:
    #    print(stu[0],stu[2])
        categories.append([stu[0].strip(),stu[2].strip()])
        
    categories1 = categories[1:]
#    cd = r'D:\zhu\chen1\data\pick5\p_app_resize_Td_sts'
    p5_apps = []
    for a in os.listdir(cd):
        app_dir = os.path.join(cd,a)
        a_ui_num = 0
        for b in os.listdir(app_dir):
            a_ui_num += 1
        p5_apps.append([a.strip(), a_ui_num])
    
    categories3 = []
    for c in categories1:
        if c[0] in [p[0] for p in p5_apps]:
            c3 = [d for d in p5_apps if d[0]==c[0]][0]
            categories3.append([c[0], c[1], c3[1]])
    
    if app:        
        # app数量
        p5_c_l = []
        p5_c_d = {}
        for c in categories3:
            if c[1] not in p5_c_l:
                p5_c_l.append(c[1])
                p5_c_d[c[1]] = [c[0]]
            else:
                p5_c_d[c[1]] += [c[0]]
                
#        p5_c_ds = sorted(p5_c_d.items(), key = lambda x:x[1])
        return p5_c_l, p5_c_d
    else:
        # ui数量
        p5_c_l1 = []
        p5_c_d1 = {}
        for c in categories3:
            if c[1] not in p5_c_l1:
                p5_c_l1.append(c[1])
                p5_c_d1[c[1]] = int(c[2])
            else:
                p5_c_d1[c[1]] += int(c[2])
                
        p5_c_ds1 = sorted(p5_c_d1.items(), key = lambda x:x[1])
        return p5_c_l1, p5_c_ds1

def get_subsidiary(cd):
    apps_l = []
    subsidiary_l = []
    subsidiary_d = {}
    for a in os.listdir(cd):
        app_dir = os.path.join(cd,a)
        a_ui_num = 0
        for b in os.listdir(app_dir):
            a_ui_num += 1
        apps_l.append([a.strip(), a_ui_num]) # app名，ui数量
        
        if a.split('.')[0] == 'com':
            subsidiary = a.split('.')[1]
        else:
            subsidiary = a.split('.')[0]
            
        if subsidiary not in subsidiary_l:
            subsidiary_l.append(subsidiary)
            subsidiary_d[subsidiary] = [a]
        else:
            subsidiary_d[subsidiary] += [a]
        subsidiary_d1 = sorted(subsidiary_d.items(), key = lambda x:len(x[1]), reverse=True)
    return subsidiary_l, subsidiary_d

#-------------------main start-----------------------------------
if __name__ == "__main__":
    cd = r'.\p_app_resize_Td_sts'
    ui_dictionary = load_dataset(cd)    

#    # write the selected apps to the file
    _file = r'.\data\data.txt'
    train_apps, valid_apps, test_apps = split_valid_datasets(ui_dictionary)    
    for app in train_apps:
        write_file(_file, app+',')
    write_file(_file, ';')
    for app in valid_apps:
        write_file(_file, app+',')
    write_file(_file, ';')
    for app in test_apps:
        write_file(_file, app+',')
        
    # read the selected apps from the file
    train_apps, valid_apps, test_apps = read_valid_file(_file)
    
#    # 获取随机选定好的ui， 并将它们存入txt文件
    _train_file = r'.\data\train_ui.txt'
    _valid_file = r'.\data\valid_ui.txt'
    _test_file =  r'.\data\test_ui.txt'
    
    train_uis = get_random_ui(ui_dictionary,train_apps)
    valid_uis = get_random_ui(ui_dictionary,valid_apps)
    test_uis = get_random_ui(ui_dictionary,test_apps)
    
    # write the pair data to file        
    for ui in train_uis:
        write_file(_train_file, ui+',')
    for ui in valid_uis:
        write_file(_valid_file, ui+',')
    for ui in test_uis:
        write_file(_test_file, ui+',')
     #------------------------------------------------------#  
#    # read the pair data from file
    train_uis = read_file(_train_file) # 6368
#    valid_uis = read_file(_valid_file) # 398
#    test_uis = read_file(_test_file) # 732
    
    