# -*- coding: utf-8 -*-
"""
(1)批量保留subtree_info在txt,取得structural信息，并且给每个subtree编号
(2)save_subtree_img in png file
(3)del ori ui
(4)resizing
"""

import os,shutil
import numpy as np
from get_subtree import get_component_byjson, get_className, get_layerNum, get_width, get_height, get_resized_bounds
from Comm_utils import compressImage

def get_clayer_nodes(Dom_list,layer1_ids=[]): # 只用来找第一层了
    print('layer1_ids:',layer1_ids)
    Dom_list1 = Dom_list.copy() # 只需要传入Dom_list
    #存放筛选本层兄弟节点之后剩余的，非本层节点也非本层节点的兄弟节点
    # 开始找第一层所有node
    Dom_listids = [c[0] for c in Dom_list1]
    r = np.min([get_layerNum(c) for c in Dom_listids])
    print('\nr:', r)
    # layer1_ids存第一次搜索最短的所有节点
    layer1_ids += [c for c in Dom_listids if get_layerNum(c)==r and c not in layer1_ids]
    print('layer1_ids:',layer1_ids)
    # 删除第一层nodes
    for c in Dom_list1:
        if c[0] in layer1_ids:
            Dom_list1.remove(c)
    # 删除第一层nodes的children
    NC_dict = {}
    for c in Dom_list:
#        print('\n\nc[0]', c[0])
        for f in layer1_ids:
#            print('\nf',f)
            if f in c[0] and f!=c[0]: #找到第一层的所有孩子
                if c[0] not in NC_dict.keys():
                    NC_dict[f] = [c]
#                    print(NC_dict)
                else:                    
#                    print('c',c)
                    NC_dict[f].append(c)
#                    print(NC_dict)
                if c in Dom_list1:
                    Dom_list1.remove(c)
    if len(Dom_list1)==0:
        print('\n',layer1_ids,'no more layer1 Node')
        return layer1_ids
    else:
        print('\nExtra',len(Dom_list1) ,'layer1 Node:')
        [print(c[0]) for c in Dom_list1] # 再找它们的children
        _ids = get_clayer_nodes(Dom_list1,layer1_ids)
        return _ids

def get_subLeafNodes_ofC(c,Dd,Dl): # 找Dl中只属于c的项不属于Dd中任何其他的leaf
    Dd_ids = [d[0] for d in Dd]
    Dl_ids = [l[0] for l in Dl]
    leaf_ids_inDd = []
    for d in Dd_ids:
        for l in Dl_ids:
            if d in l:
                if l not in leaf_ids_inDd:
                    leaf_ids_inDd.append(l)
    leaf_ids_offDd = [l for l in Dl_ids if l not in leaf_ids_inDd]
    c_leaf = []
    for l in leaf_ids_offDd:
        if c in l:
            if l not in c_leaf:
                c_leaf.append(l)
    return c_leaf

def get_subNodes_ofC(c,Dd): # 找到Dd中只属于c的node
    Dd_ids = [d[0] for d in Dd]
    c_children = []
    for d in Dd_ids:
        if c in d:
            if d not in c_children:
                c_children.append(d) # c的所有孩子
    # 找只属于c的
    others_subNodes = []
    for d in c_children: # 判断d是否存在其他父节点
        c_children1 = c_children.copy()
        c_children1.remove(d)
        for f in c_children1: # 在除d之外其他Dd中遍历
            if f in d:
                if d not in others_subNodes:
                    others_subNodes.append(d)
    c_children = [c for c in c_children if c not in others_subNodes]
    return c_children

def write_NtD(c,node_tree,Dd,Dl): # c不在Dd中，把node的list写入dict，
    #有多少subNodes，dict有多少想，(k节点编号,v只属于它的childre的节点编号)
    # (1)找c的leaf(只属于c)
    c_leafs= get_subLeafNodes_ofC(c,Dd,Dl)
    Dl_out = [d for d in Dl if d[0] not in c_leafs] # 输出
    
    # (2)找c的subnode(只属于c)
    c_childrens = get_subNodes_ofC(c,Dd)
    Dd_out = [d for d in Dd if d[0] not in c_childrens]#输出
    
    c_nodes_leaf = c_childrens + c_leafs
    node_tree[c] = c_nodes_leaf # 更新node_tree
    
    if c_childrens == []:
        return node_tree,Dd_out,Dl_out
    else:
        for c1 in c_childrens:
            node_tree1,Dd1,Dl1 = write_NtD(c1,node_tree,Dd_out,Dl_out)
        return node_tree1,Dd1,Dl1

def DtoT(c,tt,node_tree,id_class_d): # 递归写树结构
    if c not in node_tree.keys():
        tt+='}'
        return tt
    else:
        for d in node_tree[c]:
#            print('\nd:',d, id_class_d[d])
            tt+='{'+id_class_d[d] # 写入class 真实需要的
#            tt+='{'+d             # 写入node_id，测试用
            tt = DtoT(d,tt,node_tree,id_class_d)
        tt+='}'
    return tt

def get_aptedTree(Dom_list,node_tree,id_class_d):#获取DomTree和Dom_list中所有node的subtree结构
    aTree = []
    # root 节点
    if 'root' in [c[0] for c in Dom_list]:
        tree_text = '{root'
        tree_text = DtoT('root',tree_text,node_tree,id_class_d)
    #    tree_text+='}'
        c_b = [c[2] for c in Dom_list if c[0]=='root'][0]
        aTree.append(['root:',tree_text,c_b])
    #    aTree['root'] = tree_text
    
    # subtrees 节点
    for c in Dom_list:
        tree_text = '{'+ c[1]
        tree_text = DtoT(c[0],tree_text,node_tree,id_class_d)
#        tree_text+='}'
        aTree.append([c[0]+':',tree_text,c[2]])
    
    return aTree

def save_subtree_img(_list):
    if _list != []:
    for c in _list:# 去掉高度过半的subtree
        if get_height(c[2]) >= 1280:
            _list.remove(c)                    
    
    for c in _list:        
        resized_block = get_resized_bounds(c[2])
        cropedIm = im.crop(resized_block)
        save_dir = os.path.join(dst_dir,str(ui_id))
        if(not os.path.exists(save_dir)):
            os.makedirs(save_dir)
        shutil.copy(ui_dir, save_dir)
        save_subtree_dir = os.path.join(save_dir,c[0])
        cropedIm.save(save_subtree_dir+'.png',"png")

'''save subtree info in .txt'''
def save_tree_process(jsns_dir,ui_id,tree_dir,db_app_dir):
    cd0  = os.path.join(jsns_dir,ui_id+'.json')
    
    Dom_list,leaf_list,st_bank = get_component_byjson(cd0)# 所有叶子节点均有father在Dom_list
    
    save_subtree_img(Dom_list)
    
    if Dom_list != []:                
        if Dom_list==[]:
            print(ui_id, 'Dom_list is Empt!')
        else:
            print('Dom_list.len:', len(Dom_list))
            print('st_bank.len: ', len(st_bank))
            Dom_list = [(c[0],get_className(c[1]),c[2]) for c in Dom_list]
            leaf_list = [(c[0],get_className(c[1]),c[2]) for c in leaf_list]
            Dom_allnodes = Dom_list+leaf_list  
            
            #构件Dom_list中每个节点的子节点(包括leaf节点)    
            node_tree = {}                      # p1 需要写和返回的dict
            dynamic_Dom_list = Dom_list.copy()  # p2 Dom_list还剩几个没写入的node
            dynamic_leaf_list = leaf_list.copy()# p3 leaf_list还剩几个没写入的leaf
            
            # 开始
            print('layer',len(node_tree),'root')
            print('\ndynamic_Dom_list', dynamic_Dom_list)
            layer1_ids = get_clayer_nodes(dynamic_Dom_list,[]) # 获得第一层结果
            dynamic_Dom_list = [c for c in dynamic_Dom_list if c[0] not in layer1_ids]
            node_tree['root'] = layer1_ids
            print('\n\n',len(layer1_ids),'first layers!')
            for c in layer1_ids: # 层第一层节点往下延申        
                node_tree,dynamic_Dom_list,dynamic_leaf_list = write_NtD(c,node_tree,dynamic_Dom_list,dynamic_leaf_list)
            
            node_tree1 = node_tree.copy()
            for (k,v) in node_tree.items():
                if k not in [c[0] for c in Dom_list]:
                    node_tree1.pop(k)
                
        # 用node_tree开始构建形如tree1=Tree.from_text('{A{B{M}{N}{F}}{C}}')
            id_class_d = {} # 构建一个所有节点id-class的dict
            for c in Dom_allnodes:
                id_class_d[c[0]] = c[1]
            
    #        aTrees = get_aptedTree(Dom_list,node_tree,id_class_d)
            aTrees = get_aptedTree(Dom_list,node_tree1,id_class_d)
            ui_tree_dir = os.path.join(tree_dir,ui_id+'.txt')            
            
            f1 = open(ui_tree_dir,'w',encoding='utf-8') # width, height
#            [f1.write(t[0]+t[1]+':'+str(get_width(t[2]))+'_'+str(get_height(t[2]))+',\n') for t in aTrees]
            [f1.write(t[0]+t[1]+':'+str(get_width(t[2]))+'_'+str(get_height(t[2]))+':'+str(t[2])[1:-1]+',\n') for t in aTrees] # 再写入起始坐标   
            f1.write(';\n') # 写node_tree dict
            for (k,v) in node_tree1.items():
                f1.write(str(k)+' '+str(v)+'\n')
            f1.close()
        # save st_bank_in
        ui_db_dir = os.path.join(db_app_dir,ui_id+'.txt')
        f2 = open(ui_db_dir,'w',encoding='utf-8')
        [f2.write(c+',\n') for c in st_bank]

def remove_empty_f(apps_dir):
    for app in os.listdir(apps_dir):
        app_dir = os.path.join(apps_dir,app)
        if not os.listdir(app_dir): # 判定为empty folder
            print('Empty',app_dir) 
            os.rmdir(app_dir)
            
#-------------------main start-----------------------------------
# save_tree_process(jsns_dir,ui_id,tree_dir):

if __name__ == "__main__":   
    jsns_dir = r'.\Rico\combined' # json_dir
    cd = r'.\P_app_resize_sub' # GUI_dir
    dt = r'.\aTrees_dict_app' # ourput_dir
    db = r'.\st_bank_app' #   # save id_list with blank
    dsts_dir = r'.\p_app_resize_Td_sts' # save subtree imgs
    
    for app in os.listdir(cd):
        app_dir = os.path.join(cd,app)
        dt_app_dir = os.path.join(dt,app)
        db_app_dir = os.path.join(db,app)
        dst_dir = os.path.join(dsts_dir,app)
        if not os.path.exists(dt_app_dir):
            os.makedirs(dt_app_dir)
        if not os.path.exists(db_app_dir):
            os.makedirs(db_app_dir)
        for im in os.listdir(app_dir):
            img_dir = os.path.join(app_dir,im)
            ui_id = os.path.splitext(im)[0]
            
            save_tree_process(jsns_dir,ui_id,dt_app_dir,db_app_dir)
            
            ori_img = os.path.join(os.path.join(app_dir,im),im+'.jpg')
            if os.path.exists(ori_img):
                os.remove(ori_img)
            else:
                print('no ori_img:', ori_img)
            
    remove_empty_f(dt)
    remove_empty_f(db)
    
    resized_dir = r'.\p_app_resize_Td_sts_c_resized'
    compressImage(dsts_dir,resized_dir,512,256)
    
    
    
    
    
    
            