# -*- coding: utf-8 -*-

import os,random,time,csv
import numpy as np
from apted import APTED, Config
from  apted.helpers import Tree

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

def write_file(path_file_name,_str):
    if not os.path.exists(path_file_name):
        with open(path_file_name, "w") as f:
            print(f)
    with open(path_file_name, "a") as f:
        f.write(_str)
        
def read_valid_file(_file):
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
        print('random select: ', arr)
        arr = arr[:n_max]
        print('random select cut: ', arr)
        for j in range(int(len(arr))):
            selected_uis.append(current_app_dirs[arr[j]])
            print('selected ui:', current_app_dirs[arr[j]])
    return selected_uis
  
# read UI_pair
def read_file(ui_file): # "；" is used to split intra pair and inter pair
    _uis = []
    with open(ui_file, 'r') as f:
        for line in f.readlines():
            linelist = line.split(',')
            _uis.append(linelist)
    return _uis[0][:-1]

#-------------------main start-----------------------------------
if __name__ == "__main__":
    cd = r'.\p_app_Td_sts'
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
    
    
