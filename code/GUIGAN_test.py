# -*- coding: utf-8 -*-

import os,time,gc,random,math,csv,argparse
import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn import (manifold, datasets, decomposition, ensemble,random_projection, metrics)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import kneighbors_graph
from scipy.stats import mode

from GUIGAN_main import generate_samples
from generator import Generator
from get_style_emb import read_data,get_ui_info
from comm import get_bank_size,get_Repository,get_list_wbk

import sys
sys.path.append(r'.\StyleEmbedding')
from load_data import get_s_app

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=0, type=int)
opt = parser.parse_args()
print(opt)

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 32
# TOTAL_BATCH = 200
TOTAL_BATCH = 30

POSITIVE_FILE = 'real' 
EVAL_FILE = 'eval' 

if opt.cuda is not None and opt.cuda >= 0:
    print('opt.cuda is not None')
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True
else:
    print('opt.cuda is None')

# Genrator Parameters
g_emb_dim = 32      # Embedding 
g_hidden_dim = 32   # nn.LSTM(emb_dim, hidden_dim)
g_sequence_len = 30

# Discriminator Parameters
d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
d_dropout = 0.75
d_num_class = 2 

bank_dict = {'1':2, '2':6, '3':10, '4':20, '5':35, '6':50, '7':70, '8':100, '9':200, '10':300}
pre_built = False

if __name__ == '__main__':       
    
    random.seed(SEED)
    
    file_csv = r'F:\2017\zhu\RicoDataset\app_details.csv'    
    cd_img = r'D:\zhu\chen1\data\pick8\p_app_resize_Td_sts_c_resize512_noui1'
    txt_img = r'D:\zhu\chen1\data\pick8\aTrees_dict_app'
    st_dir = r'D:\zhu\chen1\data\pick8\p_app_resize_Td_sts'
    db_dir = r'D:\zhu\chen1\data\pick8\st_bank_app'
    emb_file = r'D:\SpyderWorkSpace\GUI_Ganerator\data\categories_app_emb'
    
    _save_path = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\model'    
    if pre_built:
        NEGATIVE_FILE = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\results_test1' 
    else:
        NEGATIVE_FILE = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\results_test_pre-built'
        
    #---------------------------
    file_csv = r'app_details.csv'  
    cd_img = r'.\p_app_Td_sts_resized'    
    txt_img = r'.\aTrees_dict_app'  
    st_dir = r'.\p_app_Td_sts'
    db_dir = r'.\st_bank_app'
    emb_file = r'.\data\categories_app_emb'
    
    m_save_path = r'.\models' # 3 loss    
    if pre_built:
        NEGATIVE_FILE = r'.\results' 
    else:
        NEGATIVE_FILE = r'.\results_pre' 
    
    appsl, appsd = get_s_app(file_csv, st_dir)
    appsl1 = []
    for (k,v) in appsd.items():
        appsl1.append([k,len(v)])
    appsl1 = sorted(appsl1, key=lambda x: x[1], reverse=True)
    
    _ns = []
    _n = 0; _ns.append(_n) # News & Magazines
    
    c_apps = []
    c_cats = []
    for _n in _ns:
        c_cat = appsl1[_n][0]
        c_cats.append(c_cat) 
        c_apps += appsd[c_cat]    
    print('\nc_cats: ', c_cats)
    
    real_data_bk = get_list_wbk(db_dir)
    real_data_bk_c = [c for c in real_data_bk if c[0] in c_apps]

    starttime = time.time()
    x_ids = []
    x_emb = []
    c_cat = ''
    for _cat in c_cats:        
        c_cat_emb_file = os.path.join(emb_file, str(_cat)+'.txt')
        _x_ids,_x_emb = read_data(c_cat_emb_file) 
        x_ids += _x_ids
        if x_emb == []:
            x_emb = _x_emb
            c_cat = _cat
        else:
            x_emb = np.concatenate((x_emb, _x_emb), axis = 0)
            c_cat = c_cat + '_and_' + _cat
    
    endtime = time.time(); dtime = endtime - starttime    
    print("\nTime for loading training embedding：%.8s s" % dtime)
        
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
    
    '''load dom_tree'''    
    train_uis1 = []
    for app1 in os.listdir(txt_img): # (2) subtree txt dir
        if app1 in c_apps:
            app_dir1 = os.path.join(txt_img,app1)
            for ui1 in os.listdir(app_dir1):
                ui_dir1 = os.path.join(app_dir1,ui1)
                train_uis1.append(ui_dir1)
                
    train_uis_tree,train_templates_list1,train_templates_dict1 = get_Repository(train_uis1)
    train_DT = []
    train_DT1 = []
    train_uis_tree = sorted(train_uis_tree,key=lambda x: x[0].split('.txt')[0].split('_')[-1])
    for ui in train_uis_tree:
        s = ''
        if len(ui[1]) == 1:
            continue        
        for (k,v) in ui[1].items():
            s += v
        train_DT.append(s)
        s1 = ''
        s2 = []
        ui_sorted = sorted(ui[1].items(),key=lambda x: (len(x[0]), x[0])) 
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
        
    start_id_list = [] 
    end_id_list = []
    
    starttime = time.time()
    real_data_id = []
    real_data = []

    x_ids2 = [[x[0], x[4], st_dir+'\\'+os.path.basename(os.path.dirname(x[4]))+'\\'+os.path.basename(x[4]).split('_')[0]+'\\'+
               os.path.basename(x[4]).split(os.path.basename(x[4]).split('_')[0]+'_')[-1],x[5]] for x in x_info]
    uis = [x[0].split('_')[0] for x in x_ids2]
    uis = list(set(uis))
    uis = sorted(uis,key=lambda x: x) 
    fit_ui_banks = []
    real_world_data = []
    for ui in uis: 
        fit_uis = [x for x in x_ids2 if ui == x[0].split('_')[0]]     
        fit_uis = sorted(fit_uis,key=lambda x: (int(x[3].split(',')[1]), 1/(int(x[3].split(',')[3])-int(x[3].split(',')[1]))))
        fit_uis_bk = [x for x in real_data_bk_c if ui == x[1]][0][-1]
        fit_ui_banks.append(fit_uis_bk)
        st_id_list = []
        st_list = []
        for u in fit_uis:
            st_id_list.append(x_ids2.index(u)+11)
            st_list.append(u)            
            if u == fit_uis[-1]:
                continue            
            st_id = u[0][len(u[0].split('_')[0])+1:] 
            _id = fit_uis_bk.index(st_id)                              
            for i in range(len(fit_uis_bk)):
                st = fit_uis_bk[_id+i+1]
                if st.split('_')[0] == 'bk':
                    bk = get_bank_size(int(st.split('_')[1]))
                    st_id_list.append(bk[0])       
                    st_list.append(bk)
                else:
                    break
            
        if len(st_list) == 1: 
            continue
        real_data_id.append(st_id_list)
        real_data.append(st_list)
        start_id_list.append(st_id_list[0])
        end_id_list.append(st_id_list[-1])
    
    real_data_id0 = real_data_id.copy()
    real_data_id = [x[:g_sequence_len] for x in real_data_id] 
    real_data_id1 = [np.pad(x, (0,g_sequence_len - len(x))) for x in real_data_id]
    endtime = time.time(); dtime = endtime - starttime
    print("\nTime for loading real world data：%.8s s" % dtime)
        
    GENERATED_NUM = len(real_data_id1)
    print('\nGENERATED_NUM,real_data_id1', GENERATED_NUM)
        
    VOCAB_SIZE = len(x_info_ids)+1+10 
    print('\nVOCAB_SIZE:',VOCAB_SIZE)    
    print('real_vocab_size: ', len(x_info_ids))
    
    starttime = time.time()
    x_index = []
    for i in range(len(x_ids)):
        if x_ids[i] not in x_info_ids:
            x_index.append(i)
    
    x_ids_not_in_info = [x_ids[i] for i in x_index]
    x_ids = [x_ids[i] for i in range(len(x_ids)) if i not in x_index]
    
    x_emb_not_in_info = [x_emb[i] for i in x_index]
    x_emb = [x_emb[i] for i in range(len(x_emb)) if i not in x_index]
    endtime = time.time(); dtime = endtime - starttime    
    print("\nTime for unifying x_info and x_emb：%.8s s" % dtime)
    print('\n')   
           
    reduced_data1 = PCA(n_components=2).fit_transform(x_emb)
  
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    
    _save_path = os.path.join(_save_path, c_cat)
    _save_path = os.path.join(_save_path, 'generator'+str(TOTAL_BATCH-1)+'.pkl')

    generator.load_state_dict(torch.load(_save_path))
    
    if opt.cuda:
        generator = generator.cuda()
    
    start_st = random.sample(start_id_list, BATCH_SIZE)    
    start_st = np.expand_dims(start_st, axis=1)
    start_st = Variable(torch.Tensor(start_st).long())
    
    # start_st1 = [start_id_list[0] for c in range(BATCH_SIZE)]
    # start_st1 = np.expand_dims(start_st1, axis=1)
    # second_st1 = np.expand_dims(second_st1, axis=1)
    # preb_st = np.concatenate((start_st1, second_st1), axis = 1)
    # preb_st = Variable(torch.Tensor(preb_st).long())
    
    NEGATIVE_FILE0 = NEGATIVE_FILE
    
    for _id in range(len(real_data_id1)):
        if (_id %5 ==0):    
            pre_len = 3
            pre_s = list(real_data_id1[_id][:pre_len])
            print(real_data[_id][0][0])
            # # start_st = [pre_s1 for c in range(5)]
            # pre_s = [pre_s for c in range(BATCH_SIZE)]
            # pre_s = Variable(torch.Tensor(pre_s).long())
            
            '''generate samples'''
            NEGATIVE_FILE0 = os.path.join(NEGATIVE_FILE, str(_id)+'_'+str(pre_len))
            if not os.path.exists(NEGATIVE_FILE0):
                os.mkdir(NEGATIVE_FILE0)
            NEGATIVE_FILE1 =  NEGATIVE_FILE0 + '\\gene'
            
            if pre_built:
                samples_lenth = generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE1,
                                              x_info,x_ids,start_id_list,end_id_list,bank_dict,pre_s)
            else
                samples_lenth = generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE1,
                                              x_info,x_ids,start_id_list,end_id_list,bank_dict)
            
            # fig0 = plt.figure(figsize=(20,4.5), dpi=135)
            # fig0.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
            
            # fig0.add_subplot(121)    
            # real_data_lenth = [len(x) for x in real_data_id]
            # real_data_lenth.sort()
            # plt.hist(real_data_lenth,10)
            # plt.legend()
            # real_data_mean = round(np.mean(real_data_lenth),2)
            # real_data_var = round(np.var(real_data_lenth),2)
            # plt.title('Real data lenth, mean:'+str(real_data_mean)+', var: '+str(real_data_var))
            
            # fig0.add_subplot(122)
            # samples_lenth.sort()
            # plt.hist(samples_lenth,10)
            # plt.legend()
            # samples_mean = round(np.mean(samples_lenth),2)
            # samples_var = round(np.var(samples_lenth),2)
            # plt.title('Samples lenth, mean:'+str(samples_mean)+', var: '+str(samples_var))
            
            # plt.show()
            
            #-------------------------------------
            '''connect subtrees'''
            from PIL import Image,ImageFont,ImageDraw
        #    from get_real_data2 import generate_img
            path1 = os.path.join(NEGATIVE_FILE0, 'geneimgdir.txt')
        #    path2 = os.path.join(NEGATIVE_FILE0, 'samples_0\\') # all
        #    path2 = os.path.join(NEGATIVE_FILE0, 'samples_1\\') # fit
            
            _resize = True
            if _resize:
                path2 = os.path.join(NEGATIVE_FILE0, 'samples_resize\\') # fit_resize
            else:
                path2 = os.path.join(NEGATIVE_FILE0, 'samples_no_resize\\') # fit_no resize
            if not os.path.exists(path2):
                os.mkdir(path2)
            path3 = os.path.join(NEGATIVE_FILE0, 'gene_e.txt')
            _n = 0
            if os.path.exists(path1):
                
                ori_st_dir = st_dir; data_file = path1; _path = path2; 
                total_num = 0
                _h_short = 0
                _c_one = 0
                c_fit = 0
                if not os.path.exists(_path):
                    os.mkdir(_path)
                with open(data_file, 'r', encoding="utf-8") as f:
                    lines = f.readlines()
                with open(path3, 'r', encoding="utf-8") as f1:
                    lines1 = f1.readlines()
                lis = []
                lis1 = []
                lis2 = []
                lis3 = []
                i = 0
                count = 0
                for line in lines:
                    l = line.strip().split(' ')
                    l1 = []
                    l2 = []
                    l3 = []
                    n = 0
                    for d in l:
                        if d in [str(x) for x in bank_dict.values()]:
                            l1.append(int(d))
                            l2.append(d)
                            l3.append('bank_'+str(d))
                        else:
                            d1 = os.path.split(d)
                            d0 = os.path.split(d1[0])
                            app = d0[1]
                            app_dir = os.path.join(ori_st_dir,app)
                            
                            _input = d1[1]
                            ui = _input.split('_')[0]
                            ui_dir = os.path.join(app_dir,ui)
                            sd = _input.split(ui+'_')[-1]
                            sd_dir = os.path.join(ui_dir,sd)
                            l1.append(sd_dir)
                            if app not in l2:
                                l2.append(app)
                                l3.append(n)
                                n += 1
                            else:
                                count += 1
                                l2.append(app)
                                l3.append(l2.index(app))
                    lis.append(l)
                    lis1.append(l1)
                    lis2.append(l2)
                    lis3.append(l3)
                    i += 1
                
                '''save img'''
                width_size = 512
                resize_h = 1024
                i = -1
                if _n !=0:
                    lis1 = lis1[:_n]
                    
                for s in range(len(lis1)):
                    s_name = ''
                    total_num += 1
                    total_h = 0
                    imghigh = len(lis1[s])
                    
                    if imghigh < 2:       
                        _c_one += 1
                        s_name = '_u2st'
                    
                    imagefile = []
                    for st in range(len(lis1[s])):
                        if lis1[s][st] in [x for x in bank_dict.values()]:
                            total_h += lis1[s][st]
                            imagefile.append(int(lis1[s][st]))
                        else:
                            sImg = Image.open(lis1[s][st])
                            w,h=sImg.size
                            total_h += h
                            dImg=sImg.resize((width_size,h),Image.ANTIALIAS)  #设置压缩尺寸和选项，注意尺寸要用括号                       
                            imagefile.append([dImg,h,lis3[s][st]])                    
                    i += 1
                    
                    if total_h < 600:
                        _h_short += 1
                        s_name += '_u800'
                        continue
                    else:
                        c_fit += 1
                    
                    _size = 2
                    target = Image.new('RGB',(width_size,total_h+_size*(imghigh-1)))#最终拼接的图像的大小
                    left = 0
                    right = 0        
                    
                    n = 0
                    font_size = 20
                    font = ImageFont.truetype('SIMYOU.TTF',font_size)
                    bank_ = False
                    for image in imagefile:            
                        if isinstance(image,int):
                            bank_ = True
                            stamp=Image.new("RGB",(width_size,image),(255,255,255)) # white bank
                            right += image
                            box = (0,left,width_size,right)
                            target.paste(stamp,box)
                            left += image  
                        else:
                            right += image[1]
                            box = (0,left,width_size,right)
                            target.paste(image[0],box)
                            left += image[1]  
                            
                        if n != len(imagefile)-1:
                            # stamp=Image.new("RGB",(width_size,_size),(255,0,0)) # red line
                            stamp=Image.new("RGB",(width_size,_size),(255,255,255)) # white line
                            right += _size
                            box = (0,left,width_size,right)
                            target.paste(stamp,box)
                            left += _size  
                        n+=1
                    if not os.path.exists(_path+str(i)+'.jpg'):
                        target_dImg=target.resize((width_size,resize_h),Image.ANTIALIAS)     
                        _b = ''
                        if bank_ is True:
                            _b = '_wb'
                            
                        if _resize: # resize
                            if lines1[i].strip() == '-1':
                                target_dImg.save(_path+str(i)+'_'+str(imghigh)+ _b + '_long_no_end_'+s_name+'.jpg',quality=100)
                            else:
                                target_dImg.save(_path+str(i)+'_'+str(imghigh)+ _b+s_name+'_end_'+lines1[i].strip()+'.jpg',quality=100)
                        else:       # no_resize                 
                            if lines1[i].strip() == '-1':
                                target.save(_path+str(i)+'_'+str(imghigh)+ _b+'_long_no_end_'+s_name+'.jpg',quality=100)
                            else:
                                target.save(_path+str(i)+'_'+str(imghigh)+ _b+s_name+'_end_'+lines1[i].strip()+'.jpg',quality=100)
            
                print('\n_total_num: ', total_num)
                print('_c_one: ', _c_one)
                print('_h_short: ', _h_short)
                print('c_fit: ', c_fit)
    
    
    
    
    