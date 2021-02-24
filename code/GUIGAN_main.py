# -*- coding: utf-8 -*-

import os,time,random,math,csv,argparse
import numpy as np
from apted import APTED, Config
from  apted.helpers import Tree
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

from generator import Generator
from discriminator import Discriminator
from reward import Rollout
from data_iter import DisDataIter
from get_style_emb import read_data,get_ui_info
from comm import get_samples,get_bank_size,remove_0,get_Repository,get_list_wbk

import sys
sys.path.append(r'.\StyleEmbedding')
from load_data import get_s_app

class GANLoss3(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator
    """
    def __init__(self):
        super(GANLoss3, self).__init__()
        self.cf1 = torch.nn.Parameter(torch.Tensor([1.55]))
        self.cf2 = torch.nn.Parameter(torch.Tensor([1.55]))
        self.cf3 = torch.nn.Parameter(torch.Tensor([1.55]))
        
    def forward(self, prob, target, reward, _loss2, _loss3):
        """
        Args:
            prob:   (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor).bool()
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss =  -torch.sum(loss)
        
        p1 = torch.exp(-self.cf1)
        p2 = torch.exp(-self.cf2)
        p3 = torch.exp(-self.cf3)
        multiloss = torch.sum(p1*(loss**2) + self.cf1, -1) + torch.sum(p2*(_loss2**2) + self.cf2, -1) + torch.sum(p3*(_loss3**2) + self.cf3, -1)

        return multiloss, loss

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
    def forward(self, prob, target, reward):
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor).bool()
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        return -torch.sum(loss)
    
def generate_samples(model,batch_size,generated_num,output_file,x_info,x_ids,start_id_list,end_id_list,bank_dict,pre_st=0):
    samples = []
    samples1 = []
    for _ in range(int(generated_num / batch_size)):
        if pre_st == 0:
            start_st = random.sample(start_id_list, batch_size)
            start_st = np.expand_dims(start_st, axis=1)
        else:   
            start_st = [pre_st for c in range(batch_size)]        
        start_st = Variable(torch.Tensor(start_st).long())        
        sample = model.sample(BATCH_SIZE, g_sequence_len, start_st).cpu().data.numpy().tolist()
        samples.extend(sample)
    samples1,samples_tree,samples_imgdir,samples0,real_DT,samples1_e,samples_lenth = get_samples(samples,x_info,x_ids,start_id_list,end_id_list,bank_dict)
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

def get_cluster_score(x_e, x_e_label):
    n = len(x_e_label) 
    v = []
    j = 0
    for m in range(n): 
        z = 0
        per_num = len(x_e_label[m])
        label_num = [os.path.basename(os.path.dirname(x)) for x in x_e_label[m]] # 有多少label
        n_class = []
        label_num1 = []
        i = -1
        for x in label_num:
            if x not in n_class:
                n_class.append(x)
                i += 1
            color = np.ones((1,), dtype=int) * i
            if len(label_num1) == 0:
                label_num1 = color
            else:
                label_num1 = np.concatenate((label_num1, color), axis = 0)        

        if per_num ==1: 
            v.append(1.0)
        else:
            k_clusters =  len(n_class)
            if(k_clusters >1):
                km = KMeans(n_clusters=k_clusters, random_state=0).fit(x_e[m])
                _clusters = km.labels_            
                labels = np.zeros_like(_clusters)
                for i in range(k_clusters):
                    mask = (_clusters == i)
                    labels[mask] = mode(label_num1[mask])[0]
                _v = float(metrics.homogeneity_score(label_num1, labels))
                v.append(np.exp(-_v))
            else:
                z += 1
                v.append(0.0) 
        j += 1
    val = np.mean(v)
    return(val)

# ================== Parameter Definition =================
parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=0, type=int)
opt = parser.parse_args()
print(opt)

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 32
TOTAL_BATCH = 200

POSITIVE_FILE = 'real' 
EVAL_FILE = 'eval'

if opt.cuda is not None and opt.cuda >= 0:
    print('opt.cuda is not None')
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True
else:
    print('opt.cuda is None')

# Genrator Parameters
g_emb_dim = 32 
g_hidden_dim = 32
g_sequence_len = 30

# Discriminator Parameters
d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
d_dropout = 0.5
d_num_class = 2

bank_dict = {'1':2, '2':6, '3':10, '4':20, '5':35, '6':50, '7':70, '8':100, '9':200, '10':300}

if __name__ == '__main__':
    
    random.seed(SEED)

    file_csv = r'app_details.csv'  
    cd_img = r'.\p_app_Td_sts_resized'    
    txt_img = r'.\aTrees_dict_app'  
    st_dir = r'.\p_app_Td_sts'
    db_dir = r'.\st_bank_app'
    emb_file = r'.\data\categories_app_emb'
    
    m_save_path = r'.\models' # 3 loss
    NEGATIVE_FILE = '.\samples'     
    
    appsl, appsd = get_s_app(file_csv, st_dir)
    appsl1 = []
    for (k,v) in appsd.items():
        appsl1.append([k,len(v)])
    appsl1 = sorted(appsl1, key=lambda x: x[1], reverse=True) 
    
    _ns = []
    _n = 0; _ns.append(_n) # News & Magazines
    
    mul_loss = True
    
    c_apps = []
    c_cats = []
    for _n in _ns:
        c_cat = appsl1[_n][0]
        c_cats.append(c_cat) 
        c_apps += appsd[c_cat]
    print('\n', c_cats)
    
    real_data_bk = get_list_wbk(db_dir)
    real_data_bk_c = [c for c in real_data_bk if c[0] in c_apps]    
    
    '''load embs'''
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
    
    # sample folder
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
        
    VOCAB_SIZE = len(x_info_ids)+1+10 # padding
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

    '''
    Build the network
    '''
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
        gen_gan_loss = GANLoss3()        
    else:
        gen_gan_loss = GANLoss()
        
    if opt.cuda:
        gen_gan_loss = gen_gan_loss.cuda()
        
    gen_gan_optm = optim.Adam([{"params":generator.parameters()},
                               {"params":gen_gan_loss.parameters()}],lr=0.05)
    
    dis_criterion = nn.NLLLoss(reduction='sum')# negative log likelihood loss
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()       
            
    for total_batch in range(TOTAL_BATCH):
        print('\nepoch=============: ', total_batch)
        starttime = time.time()
        if not os.path.exists(os.path.join(NEGATIVE_FILE,str(total_batch))):
            os.mkdir(os.path.join(NEGATIVE_FILE,str(total_batch)))
            
        for it in range(1):               
            start_st = random.sample(start_id_list, BATCH_SIZE)
            start_st = np.expand_dims(start_st, axis=1) 
            start_st = Variable(torch.Tensor(start_st).long())

            samples = generator.sample(BATCH_SIZE, g_sequence_len, start_st)
            samples1,samples_tree,samples_imgdir,samples0,real_DT,samples1_e,samples_lenth = get_samples(samples,x_info,x_ids,start_id_list,end_id_list,bank_dict)
            
            if generator.use_cuda:
                samples = samples1.cuda()                        
            if mul_loss:
                #------------loss_c--------------------#
                x_embs1 = []
                x_info_ids1 = []
                for samp in samples0:
                    embs1 = []
                    info_ids1 = []
                    for n in samp:
                        if n > 10:
                            embs1.append(reduced_data1[n-11])
                            info_ids1.append(x_info_ids[n-11])
                    x_embs1.append(embs1)
                    x_info_ids1.append(info_ids1)
                    
                c_loss = get_cluster_score(x_embs1, x_info_ids1)
                c_loss = torch.Tensor([c_loss])
                c_loss.requires_grad_()
                c_loss = c_loss.cuda()                        
                #------------loss_s--------------------#
                s_len = len(samples_tree)
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
                
                t_loss = torch.mean(torch.Tensor([_loss]))
                t_loss.requires_grad_()
                t_loss = t_loss.cuda()
                #-------------------------------------------# 
            
            # construct the input to the genrator, add zeros before samples and delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
            
            inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())                    
            targets = Variable(samples.data).contiguous().view((-1,))
            rewards = rollout.get_reward(samples, 16, discriminator)                    
            rewards = Variable(torch.Tensor(rewards))
            rewards = torch.exp(rewards).contiguous().view((-1,))
            
            if opt.cuda:
                rewards = rewards.cuda()
            prob = generator.forward(inputs)
            
            if mul_loss:
                loss, _ = gen_gan_loss(prob, targets, rewards, c_loss, t_loss)
            else:
                loss = gen_gan_loss(prob, targets, rewards) # after policy gradient
                                                
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()
        
        rollout.update_params() 

        g_save_path = os.path.join(m_save_path, c_cat)
        if not os.path.exists(g_save_path):
            os.mkdir(g_save_path)
        g_save_path = os.path.join(g_save_path, 'generator'+str(total_batch)+'.pkl')
        # torch.save(generator.state_dict(), g_save_path)
        print('mul_loss ',loss.item())
                      
        for p in range(4):
            NEGATIVE_FILE1 =  NEGATIVE_FILE + '\\' + str(total_batch) + '\\gene'
            samples_lenth = generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE1,x_info,x_ids,start_id_list,end_id_list,bank_dict)                                        
            NEGATIVE_FILEtxt = NEGATIVE_FILE + '\\' + str(total_batch) + '\\gene.txt'
            dis_data_iter = DisDataIter(real_data_id1, NEGATIVE_FILEtxt, BATCH_SIZE)
            for q in range(2):                        
                total_loss = 0.
                total_words = 0.                        
                n = 0 
                for (data, target) in dis_data_iter:
                    n+=1
                    data = Variable(data)
                    target = Variable(target)
                    if opt.cuda:
                        data, target = data.cuda(), target.cuda()
                    target = target.contiguous().view(-1) 
                    pred = discriminator.forward(data) 
                    loss = dis_criterion(pred, target) # negative log likelihood loss                            
                    total_loss += loss.item()
                    total_words += data.size(0) * data.size(1)       
                    
                    dis_optimizer.zero_grad() 
                    loss.backward()     
                    dis_optimizer.step()      
                
                dis_data_iter.reset() 
                f_loss = math.exp(total_loss/ total_words) 

                d_save_path = os.path.join(m_save_path, c_cat)
                if not os.path.exists(d_save_path):
                    os.mkdir(d_save_path)
                d_save_path = os.path.join(d_save_path, 'discriminator'+str(total_batch)+'.pkl')                        
                # torch.save(discriminator.state_dict(), d_save_path)
        print('total_d_loss ', total_loss)
        print('f_d_loss ',f_loss)
                
                
