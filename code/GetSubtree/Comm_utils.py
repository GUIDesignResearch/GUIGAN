# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import json,os,shutil
import matplotlib.pyplot as plt
from PIL import Image
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func('EarlyStopping counter: ', self.counter, 'out of', self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
#            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.trace_func('Validation loss decreased ({',self.val_loss_min,':.6f} --> {',val_loss,':.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def get_json(cd):
    with open(cd, encoding='utf-8') as f:
        line = f.read()
        d = json.loads(line)
        f.close()    
    a = d['activity']['root']
    return a

def get_DOM(list_Dom, father, layer_num, last_layer='0-0'):
    if('children' in father):
        i = 0
        next_layer_num = layer_num + 1
        for child in father['children']:
            c_layer = str(layer_num)+'-'+str(i)
            if('resource-id' in child):
                _id = child['resource-id']
            else:
                _id = ''
            if('children' in child):
                _children_num = len(child['children'])
            else:
                _children_num = 0
            list_Dom.append([last_layer+'_'+c_layer,child['class'],child['bounds'],_children_num,_id])
            i += 1
            get_DOM(list_Dom, child, next_layer_num, c_layer)

def get_DOM3(list_Dom, father, layer_num, last_layer='0-0'): # 改名了(c[0])
    if('children' in father):
        i = 0
        next_layer_num = layer_num + 1
        for child in father['children']: # i 在一个father下loop
            if(father['bounds']!=child['bounds'] and child['bounds'][2]>child['bounds'][0] and child['bounds'][3]>child['bounds'][1]):
                c_layer = str(layer_num)+'-'+str(i)
                if('resource-id' in child):
                    _id = child['resource-id']
                else:
                    _id = ''
                if('children' in child):
                    _children_num = len(child['children'])
                else:
                    _children_num = 0
                link = last_layer +'_'+c_layer
                list_Dom.append([link,child['class'],child['bounds'],_children_num,_id])
                
                i+=1
                print('next_layer_num', str(next_layer_num), '; link',str(link))
                get_DOM3(list_Dom, child, next_layer_num, link)
            else:
                print('same bounds with father',str(last_layer)+'_'+str(layer_num))
                get_DOM3(list_Dom, child, layer_num, last_layer)


def get_resized_bounds(block,rate = 8/3):
    resized_block = []
    for i in range(len(block)):
#        resized_block[i] = int(block[i])/ rate
        resized_block.append(int(block[i])/ rate)
    return resized_block

def cut_subtree_save(list_dom, im, cd, ui_id):
    for c in list_dom:
        resized_block = get_resized_bounds(c[2])
        cropedIm = im.crop(resized_block)
        save_dir = os.path.join(cd,str(ui_id))
                
        if(not os.path.exists(save_dir)):
            os.mkdir(save_dir)
        save_subtree_dir = os.path.join(save_dir,c[0])
        cropedIm.save(save_subtree_dir+'.png',"png")

def compressImage(srcPath,dstPath, width, height): #将原来路径的图像转换为width × height大小的图像
    for filename in os.listdir(srcPath):  
        print('filename:',filename)
        #如果不存在目的目录则创建一个，保持层级结构
        if not os.path.exists(dstPath):
                os.makedirs(dstPath)

        #拼接完整的文件或文件夹路径
        srcFile=os.path.join(srcPath,filename)
        dstFile=os.path.join(dstPath,filename)
        print('srcFile:',srcFile)
        print('dstFile:',dstFile)

        #如果是文件就处理
        if os.path.isfile(srcFile):
            if(os.path.splitext(srcFile)[1] == '.jpg' or os.path.splitext(srcFile)[1] == '.bmp' or os.path.splitext(srcFile)[1] == '.png'):
            #打开原图片缩小后保存，可以用if srcFile.endswith(".jpg")或者split，splitext等函数等针对特定文件压缩
                sImg=Image.open(srcFile)
#                sImg = sImg.convert("L") # 改变深度为8
                w,h=sImg.size
                dImg=sImg.resize((width,height),Image.ANTIALIAS)  #设置压缩尺寸和选项，注意尺寸要用括号
                dImg.save(dstFile) #也可以用srcFile原路径保存,或者更改后缀保存，save这个函数后面可以加压缩编码选项JPEG之类的
                print(dstFile+" compressed succeeded")

        #如果是文件夹就递归
        if os.path.isdir(srcFile):
            compressImage(srcFile,dstFile, width, height)

#-------------------main start-----------------------------------
if __name__ == "__main__":
#    cd = r'D:\zhu\chen1\data'
    output_dir = r'D:\zhu\chen1\data\cut_subtrees'
    
#    imgs_dir = r'D:\zhu\chen1\data\pick\zzh\professional'
    imgs_dir = r'D:\zhu\chen1\data\pick\jx_184\professional'
    jsns_dir = r'D:\Rico\combined'
    
    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)
    
    #ui_id = 20168
#    ui_id = 3956
    #ui_id = 3
#    cd0  = os.path.join(cd,str(ui_id)+'.json')
#    img0 = os.path.join(cd,str(ui_id)+'.jpg')
    
    ui_count = 0
    question_uis = []
    for i in os.listdir(imgs_dir):
        ui_count+=1
        ui_id = os.path.splitext(i)[0]
        img0 = os.path.join(imgs_dir,i)
        cd0  = os.path.join(jsns_dir,ui_id+'.json')
    
        im = Image.open(img0)
        a0 = get_json(cd0)
        print('\n0_root_class ''0:', a0['class'],', b:', a0['bounds'])
        ##get_children(a0,1,len(a0['children']),0) # layer, count, num
        #get_children(a0,1)
            
        print('\nget_DOM3')
        list_Dom2 = []
        get_DOM3(list_Dom2,a0,1)  
        
        list_Dom_without_samebounds2 = [d for d in list_Dom2 if d[3]!=0]     # 去掉叶节点
        if(len(list_Dom_without_samebounds2)==0 or len(list_Dom_without_samebounds2)==1):
            question_uis.append(img0)
            continue
        list_Dom_without_samebounds2.remove(list_Dom_without_samebounds2[0]) # 去掉第一个根节点
        layer_nums2 = [int(d[0].split('_')[-2].split('-')[0]) for d in list_Dom_without_samebounds2]
        final_layer2 = np.max(layer_nums2) # 找到当前最后的父节点
        # 改完名字的最后父节点，去掉最后的父节点s
        list_Dom_wsf2 = [d for d in list_Dom_without_samebounds2 if final_layer2 != int(d[0].split('_')[-2].split('-')[0])]
        
        # 去掉改名后有id和bounds重复的节点
        list_same_idbounds = []
        for c in list_Dom_wsf2:
            if([c[0],c[2]] not in list_same_idbounds):
                list_same_idbounds.append([c[0],c[2]])
                list_Dom_wsf2.remove(c)
        list_Dom_wsf2.sort()
        
        print('\n')
        [print(c[0],c[2]) for c in list_Dom_wsf2]
        
        if(list_Dom_wsf2==[]):
            question_uis.append(img0)
            continue
        
        list_leaf2 = [c for c in list_Dom2 if c[3]==0] # 找到所有叶子节点
        print('\nwe have', len(list_leaf2), 'leaves')
        
        # 用list_f作为最后的fathers
        list_f = list_Dom_wsf2.copy() # 去掉被包含于别的f节点的f节点
        for i in range(len(list_Dom_wsf2)):
            l_c = list_Dom_wsf2.copy()
            l_c.remove(l_c[i])
            for c in l_c:
                if(list_Dom_wsf2[i][0] in c[0] and list_Dom_wsf2[i] in list_f):
                    list_f.remove(list_Dom_wsf2[i])
        if(list_f==[]):
            question_uis.append(img0)
            continue
    
        leaf_in_father_num = 0
        list_leaf2_notfind = list_leaf2.copy()
        for c in list_leaf2:
            for f in list_f:
                if(f[0] in c[0]):
                    leaf_in_father_num+=1
                    print(f[0],'find in', c[0])
                    list_leaf2_notfind.remove(c)
        print('we find', str(leaf_in_father_num), 'leaves in fathers')
        [print(c[0]+'not found in fathers') for c in list_leaf2_notfind]
        
        cut_subtree_save(list_f, im, output_dir, ui_id)
        dst_path = os.path.join(output_dir,ui_id)
        dst_img = os.path.join(dst_path, '0.origin_'+str(ui_id)+'.png')
        shutil.copy(img0, dst_img)
    
    [print(q) for q in question_uis]
    
    #plt.imshow(cropedIm)
    #plt.show()