
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 22:59:45 2019
(11_3)test
@author: ztm
"""

import os,time,gc,random,math
import argparse
import tqdm
import numpy as np
import csv

import numpy as np
import matplotlib.pylab as plt

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

from generator import Generator

import sys
sys.path.append(r'D:\SpyderWorkSpace\GUI_Ganerator')
from load_data import read_valid_file, get_s_app
from load_style_emb import read_data,get_ui_info,get_3data
from Find_opt_seg_T import get_Repository,get_Repository1,get_Repository2
from GAN_main_gloss_cat_start_bank import remove_0, generate_samples3, get_samples2,get_bank_size, generate_samples4
from get_subtree_structures1 import get_list_wbk

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
#TOTAL_BATCH = 500 # 迭代次数
#TOTAL_BATCH = 30
TOTAL_BATCH = 10

POSITIVE_FILE = 'real' # real_data是由oracle model产生的real-world数据
#NEGATIVE_FILE = '.\results_test' # 是由Generator产生,用作pre-train G
#NEGATIVE_FILE = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\results_test'

EVAL_FILE = 'eval' # eval_file是由Generator产生，用来评价Generator和oracle model相似性所产生的数据
#VOCAB_SIZE = 5000 # 词向量个数，这里应该是template或者subtree个数 train_embedding=(34964,256), 实际107319 subtrees

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
d_dropout = 0.75
d_num_class = 2 # D二分类

if __name__ == '__main__':       
    
#     Adversarial Training
    random.seed(SEED)
    np.random.seed(SEED)    
    
    file_csv = r'F:\2017\zhu\RicoDataset\app_details.csv'    
    cd_img = r'D:\zhu\chen1\data\pick8\p_app_resize_Td_sts_c_resize512_noui1'
    txt_img = r'D:\zhu\chen1\data\pick8\aTrees_dict_app'
    st_dir = r'D:\zhu\chen1\data\pick8\p_app_resize_Td_sts'
    db_dir = r'D:\zhu\chen1\data\pick8\st_bank_app'
    
    _save_path = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\model'
    NEGATIVE_FILE = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\results_test1' # 不显示标号和红色边框线条 resize
    
    NEGATIVE_FILE = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\results_test_pre-built'
    
#    _save_path = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\model1'
#    NEGATIVE_FILE = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\results_test2' # 不显示标号和红色边框线条 resize
    
#    _save_path = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\model2'
#    NEGATIVE_FILE = r'D:\SpyderWorkSpace\GUI_Ganerator\GAN\results_test3' # 不显示标号和红色边框线条 resize
    
    '''按categories获取apps'''
    appsl, appsd = get_s_app(file_csv, st_dir)
    
    appsl1 = []
    for (k,v) in appsd.items():
        appsl1.append([k,len(v)])
    appsl1 = sorted(appsl1, key=lambda x: x[1], reverse=True) # 按照每个cat中app数量由高到低排序
    
    _ns = []
    # _n = 0; _ns.append(_n) # News & Magazines
    # _n = 1; _ns.append(_n) # Books & Reference
    # _n = 2; _ns.append(_n) # Shopping
#    _n = 3; _ns.append(_n) # Communication 
#    _n = 4; _ns.append(_n) # Entertainment
    _n = 5; _ns.append(_n) # Travel & Local
    # _n = 6; _ns.append(_n) # Lifestyle
#    _n = 7; _ns.append(_n) # Sports
#    _n = 8; _ns.append(_n) # Social
#    _n = 9; _ns.append(_n) # Music & Audio
#    _n = 12; _ns.append(_n) # Weather
    
    c_apps = []
    c_cats = []
    for _n in _ns:
        c_cat = appsl1[_n][0]
        c_cats.append(c_cat) # 选择第_n个cat
        c_apps += appsd[c_cat]
    
    print('\nc_cats: ', c_cats)
    
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
    real_world_data = []
    for ui in uis: # 每个real GUI
        fit_uis = [x for x in x_ids2 if ui == x[0].split('_')[0]]     
#        fit_uis = sorted(fit_uis,key=lambda x: (int(x[3].split(',')[1]), len(x[0]))) # 排序之后        
        fit_uis = sorted(fit_uis,key=lambda x: (int(x[3].split(',')[1]), 1/(int(x[3].split(',')[3])-int(x[3].split(',')[1]))))
        fit_uis_bk = [x for x in real_data_bk_c if ui == x[1]][0][-1]
        fit_ui_banks.append(fit_uis_bk)
        st_id_list = []
        st_list = []
        for u in fit_uis:# 每个real world data中的subtree
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
    
    '''统计一下bank'''
    bank_lsd2 = {'2':0, '6':0, '10':0, '16':0, '22':0, '32':0, '43':0, '53':0, '65':0, '100':0}
#    bank_dict = {'1':2, '2':6, '3':10, '4':16, '5':22, '6':32, '7':43, '8':53, '9':65, '10':100}
    bank_dict = {'1':2, '2':6, '3':10, '4':20, '5':35, '6':50, '7':70, '8':100, '9':200, '10':300}

    ''''''
        
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
    # 与原来相比，没有oracle,即target_lstm，lstm循环次数受限制于data的height累加是否超过额定值    
    # Define Networks    
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
#    discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)    
    
    # loading parameters
    
    _save_path = os.path.join(_save_path, c_cat)
    _save_path = os.path.join(_save_path, 'generator'+str(TOTAL_BATCH-1)+'.pkl')
#    print('_save_path: ', _save_path)
    generator.load_state_dict(torch.load(_save_path))
    
    if opt.cuda:
        generator = generator.cuda()
#        discriminator = discriminator.cuda()
        
#    for name, param in generator.named_parameters():
#        print('name: ',name, 'param: ',param.shape)
#    print('\n')
    
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
    
           # _id = 285
            pre_len = 3
            pre_s = list(real_data_id1[_id][:pre_len])
            print(real_data[_id][0][0])
            # # start_st = [pre_s1 for c in range(5)]
            # pre_s = [pre_s for c in range(BATCH_SIZE)]
            # pre_s = Variable(torch.Tensor(pre_s).long())
            
            # print('pre_s.shape',pre_s.shape)
            # given_len = pre_s.size(1)
            # print('given_len',given_len)
            
        #     '''generate samples'''
            NEGATIVE_FILE0 = os.path.join(NEGATIVE_FILE, str(_id)+'_'+str(pre_len))
            if not os.path.exists(NEGATIVE_FILE0):
                os.mkdir(NEGATIVE_FILE0)
            NEGATIVE_FILE1 =  NEGATIVE_FILE0 + '\\gene'
            # samples_lenth = generate_samples3(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE1,x_info,x_ids,start_id_list,end_id_list,bank_dict)
            samples_lenth = generate_samples4(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE1,
                                              x_info,x_ids,start_id_list,end_id_list,bank_dict,pre_s)
            
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
            '''拼接图像connect subtrees'''
            from PIL import Image,ImageFont,ImageDraw
        #    from get_real_data2 import generate_img
            path1 = os.path.join(NEGATIVE_FILE0, 'geneimgdir.txt')
        #    path2 = os.path.join(NEGATIVE_FILE0, 'samples_0\\') # all
        #    path2 = os.path.join(NEGATIVE_FILE0, 'samples_1\\') # fit
            
            _resize = True
        #    _resize = False
            if _resize:
                path2 = os.path.join(NEGATIVE_FILE0, 'samples_resize\\') # fit_resize
            else:
                path2 = os.path.join(NEGATIVE_FILE0, 'samples_no_resize\\') # fit_no resize
            if not os.path.exists(path2):
                os.mkdir(path2)
            path3 = os.path.join(NEGATIVE_FILE0, 'gene_e.txt')
            _n = 0
            if os.path.exists(path1):
        #        generate_img(st_dir, path1, path2, _n)
                
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
                    # 变化一下l的路径,ori_st_dir
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
                    lis3.append(l3) # 序号
                    i += 1
                
                '''保存图像'''
                width_size = 512#宽
                resize_h = 1024
                i = -1
                if _n !=0:
                    lis1 = lis1[:_n]
                    
                for s in range(len(lis1)):
                    s_name = ''
                    total_num += 1
                    total_h = 0
                    imghigh = len(lis1[s]) #获取当前文件路径下的文件个数,没啥用, 
                    
                    if imghigh < 2:       # 小于5个的就不要了
                        _c_one += 1
                        s_name = '_u2st'
            #            continue
                    
                    imagefile = []
                    for st in range(len(lis1[s])):
                        # 判断bank
                        if lis1[s][st] in [x for x in bank_dict.values()]:
        #                        print('lis1[s][st]: ', lis1[s][st])
                            total_h += lis1[s][st]
                            imagefile.append(int(lis1[s][st]))
                        else:
                            sImg = Image.open(lis1[s][st])
                            w,h=sImg.size
                            total_h += h
                            dImg=sImg.resize((width_size,h),Image.ANTIALIAS)  #设置压缩尺寸和选项，注意尺寸要用括号                       
                            imagefile.append([dImg,h,lis3[s][st]])
                    
                    i += 1
                    
        #            if total_h < 800: # 拼接之后太小的就不要了
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
                            stamp=Image.new("RGB",(width_size,image),(255,255,255)) # 加白色bank
                            right += image
                            box = (0,left,width_size,right)
                            target.paste(stamp,box)
                            left += image  #从上往下拼接，左上角的纵坐标递增
                        else:
                            right += image[1]
                            box = (0,left,width_size,right)
                            target.paste(image[0],box)
                            left += image[1]  #从上往下拼接，左上角的纵坐标递增
                        
        #                # 画label    
        #                if not isinstance(image,int):
        #                    draw = ImageDraw.Draw(target)
        #                    x,y=(width_size-font_size,left-font_size)
        #                    draw.text((x,y), str(image[2]), font=font, fill = 'black')
        #                    offsetx,offsety=font.getoffset('3')
        #                    width,height=font.getsize('3')
        #                    target=np.array(target)
        #                    out = target.transpose(0,1,2);
        #                    target = Image.fromarray(np.uint8(out));
                        
                        if n != len(imagefile)-1:
                            stamp=Image.new("RGB",(width_size,_size),(255,0,0)) # 加条红色线条
        #                    stamp=Image.new("RGB",(width_size,_size),(255,255,255)) # 加白色边框
                            right += _size
                            box = (0,left,width_size,right)
                            target.paste(stamp,box)
                            left += _size  #从上往下拼接，左上角的纵坐标递增
                        n+=1
                    if not os.path.exists(_path+str(i)+'.jpg'):
                        target_dImg=target.resize((width_size,resize_h),Image.ANTIALIAS)     
                        _b = ''
                        if bank_ is True:
                            _b = '_wb'
                            
                        if _resize:# resize
                            if lines1[i].strip() == '-1':
                                target_dImg.save(_path+str(i)+'_'+str(imghigh)+ _b + '_long_no_end_'+s_name+'.jpg',quality=100)
                            else:
                                target_dImg.save(_path+str(i)+'_'+str(imghigh)+ _b+s_name+'_end_'+lines1[i].strip()+'.jpg',quality=100)
                        else:# no_resize                 
                            if lines1[i].strip() == '-1':
                                target.save(_path+str(i)+'_'+str(imghigh)+ _b+'_long_no_end_'+s_name+'.jpg',quality=100)
                            else:
                                target.save(_path+str(i)+'_'+str(imghigh)+ _b+s_name+'_end_'+lines1[i].strip()+'.jpg',quality=100)
            
                print('\n_total_num: ', total_num)
                print('_c_one: ', _c_one)
                print('_h_short: ', _h_short)
                print('c_fit: ', c_fit)
    
    
    
    
    