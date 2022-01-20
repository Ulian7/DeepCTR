



import sys, os

sys.path.insert(0, '..')

import numpy as np
import torch, random
import os, psutil
from collections import OrderedDict
from tqdm import tqdm
import pickle, json, h5py
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models.din import DIN
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from config import C, print_config_oneline, config_update, argparse, config_oneline
from sklearn.metrics import accuracy_score,precision_score,recall_score, roc_auc_score
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return x


def get_xy_fd(flag_stage=1,group_index=0,group_num = 0 ):
    data_path = C.data_path
    if C.hdf:
        import pdb
        f = h5py.File(C.h5_data_path, 'r')
        num_user = len(set(f['uid'][()]))
        max_mid = f['mid'][()].max()
        feature_columns = [SparseFeat('user_id', num_user+1, embedding_dim=8), SparseFeat('mid', max_mid+1, embedding_dim=8)]
        feature_columns += [VarLenSparseFeat(SparseFeat('hist_mid', max_mid+1, embedding_dim=8), 50, length_name="seq_length")]
        behavior_feature_list = ['mid']
        import pdb
        stage_indices = f['stage{}'.format(flag_stage)]


        saving = True
        import pdb
        if saving:
            with open('./{}.txt'.format(C.data), 'w') as ff:
                for gi in range(group_num):
                    stage_indi = stage_indices['group{}'.format(gi)][()]
                    mid_s1 = f['mid'][()][stage_indi]
                    uid_s1 = f['uid'][()][stage_indi]
                    hist_mid_s1 = f['hist_mid'][()][stage_indi]
                    seq_length_s1 = f['seq_length'][()][stage_indi]
                    rating_s1 = f['rating'][()][stage_indi]
                    for gii in range(len(uid_s1)):
                        import pdb
                        write_str = '{}, {}, {}, {}, {}\n'.format(gi, uid_s1[gii], ";".join([str(e) for e in hist_mid_s1[gii]]), ";".join([str(e) for e in mid_s1[gii]]), rating_s1[gii])
                        ff.write(write_str)

            print('saved')
            f.close()
            assert 0 == 1


        if flag_stage == 2:
            stage_indices = stage_indices['group{}'.format(group_index)][()]
        else:
            stage_indices = stage_indices[()]

        import pdb
        mid_s1 = f['mid'][()][stage_indices]
        uid_s1 = f['uid'][()][stage_indices]
        hist_mid_s1 = f['hist_mid'][()][stage_indices]
        import pdb
        seq_length_s1 = f['seq_length'][()][stage_indices]
        rating_s1 = f['rating'][()][stage_indices]

        f.close()
        
        import pdb
        if(flag_stage!=4):
            mid_s1 = mid_s1[:, :C.neg_num+1].reshape([-1])
            def repeat_func(e):
                return np.repeat(np.expand_dims(e, axis=-1), C.neg_num+1, axis=-1)
            uid_s1 = repeat_func(uid_s1).reshape([-1])
            hist_mid_s1 = repeat_func(hist_mid_s1).transpose(0,2,1).reshape([-1, C.max_pad_length+1])
            seq_length_s1 = repeat_func(seq_length_s1).reshape([-1])
            rating_s1 = repeat_func(rating_s1)
            rating_s1[:, 1:] = 0
            rating_s1 = rating_s1.reshape([-1])
            import pdb
        else:
            mid_s1 = mid_s1[:, 0]
        import pdb

        feature_dict = {'user_id':uid_s1,'hist_mid':hist_mid_s1,'seq_length':seq_length_s1,'mid':mid_s1}
        x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
        y = rating_s1
        if(flag_stage!=4):
            return x, y, feature_columns, behavior_feature_list
        
        all_mid = set()
        id_to_histmid_list = ["padding"]  # 下标从1开始
        for  i in range(1,num_user+1):
            this_user_histmid = set(hist_mid_s1[i-1])  #不用把0去掉
            id_to_histmid_list.append(this_user_histmid)
            all_mid |=this_user_histmid
        import pdb
        return x, y, feature_columns, behavior_feature_list,all_mid,id_to_histmid_list




argparse()

group_list = ["padding"]  # 储存10个分组的id编号
model_list = ["padding"]  # 储存10个阶段的模型
id_group_dict = dict()
add_att_constraint = True
#x, y, feature_columns, behavior_feature_list = get_xy_fd(flag_stage)
device = 'cpu'
use_cuda = True
gpu_id = C.gpu_id = int(C.gpu_id)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gpu_id)
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'
# Tmp = input("Press any key to start")
flag_stage = C.flag_stage = int(C.flag_stage)
group_num = C.group_num = int(C.group_num)
assert group_num in [10, 100]
add_att_constraint = 'Y' if C.attReg else 'N'
print("This training is on Stage"+str(flag_stage))

# C.batch_size = 256 if C.flag_stage < 4 else 64

config_update()
print_config_oneline()


x, y, feature_columns, behavior_feature_list, all_mid, id_to_histmid_list  = get_xy_fd(flag_stage,group_num=group_num)


# Baseline
model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True, seed=C.seed)
s1_date = "0615"
path = './models/model_s1_group{}_{}_run1_{}.pth'.format(C.group_num, C.data, s1_date)
model.load_state_dict(torch.load(path))
f2 = open('staged_models/id_to_group{}_{}.pkl'.format(C.group_num, C.data),'rb')
id_group_index = pickle.load(f2)
f2.close()
model.compile('adagrad','binary_crossentropy', metrics=['acc',])
model.eval()
# this is part of the function evaluate
# pred_ans = model.predict(x, batch_size = C.batch_size)
# pred_by_group_s1 = [[] for _ in range(group_num)]    # arange the prediction by group
# y_by_group_s1 = [[] for _ in range(group_num)]
# for i in range(len(x['user_id'])):
#     index = id_group_index[x['user_id'][i]]
#     pred_by_group_s1[index] = pred_by_group_s1[index] + [pred_ans[i]]
#     y_by_group_s1[index] = y_by_group_s1[index] + [y[i]]
pred_by_group = [[] for _ in range(group_num)]    # arange the prediction by group
print("================overall metrics=============")

precision_count = []
recall_count = []
number_each_group = []
hit = 0
for i in range(group_num):#id_group_index是从0-9
    precision_count.append(0)
    recall_count.append(0)
    number_each_group.append(0)

auc_dict = {'pred':[],'actual':[]}
auc_res = []
# 用户真实标签
one_user_acutalres = [1]  
for i in range(100):
    one_user_acutalres.append(0)
for i in range(group_num):
    auc_dict['pred'].append([])
    auc_dict['actual'].append([])
    auc_res.append(0)
for i in tqdm(range(1,len(y)+1)):
    x_thisuser = dict()
    #predict函数没有将顺序打乱
    x_thisuser['user_id'] = np.array(x['user_id'][i-1]).repeat(101,axis=0)
    x_thisuser['mid'] = np.array([x['mid'][i-1]]).repeat(101,axis=0)
    x_thisuser['hist_mid'] = np.array([x['hist_mid'][i-1]]).repeat(101,axis=0)
    x_thisuser['seq_length'] = np.array([x['seq_length'][i-1]]).repeat(101,axis=0)
    this_histmid = id_to_histmid_list[i]
    import pdb
    this_histmid.add(x['mid'][i-1].item())  # 把当前预测的一个mid加进去
    not_click_mid = all_mid - this_histmid
    sample_list = random.sample(list(not_click_mid),100)
    x_thisuser['mid'][1:] = sample_list
    
    #储存sample_list的结果,101个,第一个是真实值
    import pdb
    pred_list= list(model.predict(x_thisuser,batch_size=128)) # 真实点过的mid预测分数
    auc_dict['pred'][id_group_index[i]].extend(pred_list)
    pred_actualmid = pred_list[0]
    pred_list.sort( reverse = True)
    #
    rank = pred_list.index(pred_actualmid)+1
    if(rank<=10): # 
        precision_count[id_group_index[i]]+=1#precision+1 #group是从0-9
        hit +=1
        #print("{0}users-hit:{1}".format(i,hit))
        recall_count[id_group_index[i]]+=1#recall加一
    number_each_group[id_group_index[i]]+=1#计算每个组人数
recall_res = [round(recall_count[i]/number_each_group[i],4) for i in range(len(recall_count))]
precision_res = [round(precision_count[i]/10/number_each_group[i],4) for i in range(len(precision_count))]

for i in range(group_num):
    for j in range(number_each_group[i]):
         auc_dict['actual'][i].extend(one_user_acutalres)  #test的actual数据
    auc_res[i] = roc_auc_score(auc_dict['actual'][i],auc_dict['pred'][i])


import pdb


torch.cuda.empty_cache()

for gi in range(group_num):
    model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True, seed=C.seed)
    path = './staged_models/model_stage2_Group{}-{}_{}.pth'.format(C.group_num, gi, C.data)
    model.load_state_dict(torch.load(path))
    f2 = open('staged_models/id_to_group{}_{}.pkl'.format(C.group_num, C.data),'rb')
    id_group_index = pickle.load(f2)
    f2.close()
    model.compile('adagrad','binary_crossentropy', metrics=['acc',])
    model.eval()

    pred_by_group = [[] for _ in range(group_num)]    # arange the prediction by group
    
    precision_count = []
    recall_count = []
    number_each_group = []
    hit = 0
    for i in range(group_num):#id_group_index是从0-9
        precision_count.append(0)
        recall_count.append(0)
        number_each_group.append(0)
    
    auc_dict = {'pred':[],'actual':[]}
    # 用户真实标签
    one_user_acutalres = [1]  
    for i in range(100):
        one_user_acutalres.append(0)
    for i in range(group_num):
        auc_dict['pred'].append([])
        auc_dict['actual'].append([])
    for i in range(1,len(y)+1):
        if id_group_index[i] != gi: continue
        x_thisuser = dict()
        #predict函数没有将顺序打乱
        x_thisuser['user_id'] = np.array(x['user_id'][i-1]).repeat(101,axis=0)
        x_thisuser['mid'] = np.array([x['mid'][i-1]]).repeat(101,axis=0)
        x_thisuser['hist_mid'] = np.array([x['hist_mid'][i-1]]).repeat(101,axis=0)
        x_thisuser['seq_length'] = np.array([x['seq_length'][i-1]]).repeat(101,axis=0)
        this_histmid = id_to_histmid_list[i]
        import pdb
        this_histmid.add(x['mid'][i-1].item())  # 把当前预测的一个mid加进去
        not_click_mid = all_mid - this_histmid
        sample_list = random.sample(list(not_click_mid),100)
        x_thisuser['mid'][1:] = sample_list
        
        #储存sample_list的结果,101个,第一个是真实值
        import pdb
        pred_list= list(model.predict(x_thisuser,batch_size=128)) # 真实点过的mid预测分数
        auc_dict['pred'][id_group_index[i]].extend(pred_list)
        pred_actualmid = pred_list[0]
        pred_list.sort( reverse = True)
        #
        rank = pred_list.index(pred_actualmid)+1
        if(rank<=10): # 
            precision_count[id_group_index[i]]+=1#precision+1 #group是从0-9
            hit +=1
            #print("{0}users-hit:{1}".format(i,hit))
            recall_count[id_group_index[i]]+=1#recall加一
        number_each_group[id_group_index[i]]+=1#计算每个组人数
    # recall_res = [round(recall_count[i]/number_each_group[i],4) for i in range(len(recall_count))]
    # precision_res = [round(precision_count[i]/10/number_each_group[i],4) for i in range(len(precision_count))]
    
    i = gi
    for j in range(number_each_group[i]):
         auc_dict['actual'][i].extend(one_user_acutalres)  #test的actual数据
    tmp_auc = roc_auc_score(auc_dict['actual'][i],auc_dict['pred'][i])


    print('Group: {}; AUC_base: {}; AUC: {}; Number of Each Group: {}'.format(gi, auc_res[gi], tmp_auc, number_each_group[gi]))

    torch.cuda.empty_cache()

        




        
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
