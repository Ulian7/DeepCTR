import sys

sys.path.insert(0, '..')

import numpy as np
import torch
import os
import psutil
import random
import pickle
from tqdm import tqdm
from collections import OrderedDict
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models.din import DIN
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return 

def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)
   
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print('{} memory used: {} MB'.format(hint, memory))


def get_xy_fd(flag_stage=1,group_index=0,group_num = 0 ):
#     feature_columns = [SparseFeat('user', 3, embedding_dim=8), SparseFeat('gender', 2, embedding_dim=8),
#                        SparseFeat('item', 3 + 1, embedding_dim=8), SparseFeat('item_gender', 2 + 1, embedding_dim=8),
#                        DenseFeat('score', 1)]

#     feature_columns += [VarLenSparseFeat(SparseFeat('hist_item', 3 + 1, embedding_dim=8), 4, length_name="seq_length"),
#                         VarLenSparseFeat(SparseFeat('hist_item_gender', 2 + 1, embedding_dim=8), 4, length_name="seq_length")]
#     behavior_feature_list = ["item", "item_gender"]
#     uid = np.array([0, 1, 2])
#     ugender = np.array([0, 1, 0])
#     iid = np.array([1, 2, 3])  # 0 is mask value
#     igender = np.array([1, 2, 1])  # 0 is mask value
#     score = np.array([0.1, 0.2, 0.3])
#     hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
#     hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])
#     behavior_length = np.array([3, 3, 2])
    import pickle
    if(group_num==10):
        #f2 = open('../ml-1M/temp_sample.pkl','rb')
        f2 = open('../ml-1M/temp_all_10groups.pkl','rb')
    elif(group_num == 100):
        f2 = open('../ml-1M/temp_all_100groups.pkl','rb')
    else:
         raise ValueError("Must input a group num either 10 or 100!")
    
    uid = pickle.load(f2)
    mid = pickle.load(f2)
    rating = pickle.load(f2)
    show_memory_info('before load hist_mid')
    hist_mid= pickle.load(f2)
    show_memory_info('after load hist_mid')
    seq_length = pickle.load(f2)
    stage1= pickle.load(f2)
    stage2= pickle.load(f2)
    stage3= pickle.load(f2)
    stage4= pickle.load(f2)
    f2.close()
    uset = set(uid)
    num_user = len(uset)
    pad_len = 1
    for i in range(len(hist_mid)):
        if pad_len < len(hist_mid[i]):
            pad_len = len(hist_mid[i])
    # 将hist_mid补齐为 len(hist_mid) x pad_len的矩阵
    feature_columns = [SparseFeat('user_id', num_user+1, embedding_dim=8), SparseFeat('mid',max(mid)+1
, embedding_dim=8)]
    feature_columns += [VarLenSparseFeat(SparseFeat('hist_mid', max(mid)+1, embedding_dim=8), 50, length_name="seq_length")]
    behavior_feature_list = ['mid']
    uid_s1 = []
    hist_mid_s1 = []
    mid_s1 = []
    seq_length_s1=[]
    rating_s1=[]
 #   feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
 #                   'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'score': score,
 #                   "seq_length": behavior_length}
    if flag_stage==2:
        stage1 = stage2[group_index]
    if flag_stage==3:
        stage1 = stage3
    if flag_stage==4:
        stage1 = stage4
    # import pdb
    # pdb.set_trace()
    for i in range(0,len(stage1)):
        index = stage1[i]
        uid_s1.append(uid[index])
        mid_s1.append(mid[index])
        hist_mid_s1.append(hist_mid[index])
        seq_length_s1.append(len(hist_mid[index]))
        rating_s1.append(rating[index])
    hist_mid_s1 = pad_sequences(hist_mid_s1,maxlen = pad_len+1,padding = 'pre')
    uid_s1 = np.array(uid_s1)
    hist_mid_s1 = np.array(hist_mid_s1)
    seq_length_s1 = np.array(seq_length_s1)
    mid_s1 = np.array(mid_s1)
    rating_s1 = np.array(rating_s1)
    feature_dict = {'user_id':uid_s1,'hist_mid':hist_mid_s1,'seq_length':seq_length_s1,'mid':mid_s1}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
        #y = np.array([1, 0, 1])
    x['hist_mid'] = x['hist_mid'][:,-50:]
        
    x['seq_length'] = np.where(x['seq_length']>=50,50,x['seq_length'])
    y = rating_s1
    ####################################以下代码用于存下一个list,list[1]表示uid=1的用户点过的所有item,它是一个set###########################
    if(flag_stage==4):
        all_mid = set()
        id_to_histmid_list = ["padding"]  # 下标从1开始
        for  i in range(1,num_user+1):
            this_user_histmid = set(hist_mid_s1[i-1])  #不用把0去掉
            id_to_histmid_list.append(this_user_histmid)
            all_mid |=this_user_histmid
        # f1 = open("id_to_histmid.pkl","wb")
        # pickle.dump(all_mid,f1);
        # pickle.dump(id_to_histmid_list,f1);
        # f1.close()
        #import gc
        #gc.collect()
    

        


    ######################################################end######################################################################
   
    if(flag_stage!=4):
        return x, y, feature_columns, behavior_feature_list
    else:
        return x, y, feature_columns, behavior_feature_list,all_mid,id_to_histmid_list


if __name__ == "__main__":

    group_list = ["padding"]  # 储存10个分组的id编号
    model_list = ["padding"]  # 储存10个阶段的模型
    id_group_dict = dict()
    add_att_constraint = True
    #x, y, feature_columns, behavior_feature_list = get_xy_fd(flag_stage)
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    Tmp = input("Press any key to start")
    flag_stage = int(input("Please input the Stage Number:"))
    group_num = int(input("Please input the Group number:"))
    add_att_constraint = input("Do you want to add group attention constraint?[Y/N]").upper()
    print("This training is on Stage"+str(flag_stage))

    if(flag_stage==2):
        print("There are {0} Groups in total".format(group_num))
        
        for i in range(0,group_num):
            print("Training on Stage2-Group"+str(i))
            show_memory_info('before get data')
            x, y, feature_columns, behavior_feature_list = get_xy_fd(flag_stage,i,group_num)
            show_memory_info('aftergetdata')

            model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
            model.compile('adagrad', 'binary_crossentropy',
                  metrics=['acc']) 
            model.load_state_dict(torch.load("staged_models/model_10groups.pth"))
            for param in model.parameters():
                 param.requires_grad = False
            for param in model.attention.parameters():
                param.requires_grad = True
            dict_id = list(set(x['user_id']))
            dict_group = [i]*len(dict_id)
            id_group_dict.update(dict(zip(dict_id,dict_group)))
################################################store the id_to_group_dictionary

           
            history = model.fit(x, y, batch_size=256, epochs=10, verbose=2, validation_split=0) 
            path = "staged_models/model_stage2_Group"+str(group_num)+"-" + str(i)+".pth"
            torch.save(model.state_dict(), path)
            print("Stage: "+str(flag_stage)+"   Stored in : "+path)
            print("\n")
        f1 = open('staged_models/id_to_group{}.pkl'.format(group_num),'wb')
        pickle.dump(id_group_dict,f1)
        f1.close()
        # import pdb
        # pdb.set_trace()

    if flag_stage == 1:
        x, y, feature_columns, behavior_feature_list = get_xy_fd(flag_stage,group_num=group_num)
        model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
        model.compile('adagrad', 'binary_crossentropy',
                  metrics=['auc'])
        history = model.fit(x, y, batch_size=256, epochs=10, verbose=2, validation_split=0)
        if(group_num==100):
            path = "staged_models/model_100groups.pth"
        if(group_num == 10):
            path = "staged_models/model_10groups.pth"
        #path = "staged_models/model_100groups.pth"
        torch.save(model.state_dict(), path)
        print("Stage: "+str(flag_stage)+"   Stored in : "+path)
        print("\n")
    if flag_stage == 3:
        x, y, feature_columns, behavior_feature_list = get_xy_fd(flag_stage,group_num=group_num)
        
        model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
        model.compile('adagrad','binary_crossentropy', metrics=['acc'])
        # load 10 models in stage2
        if add_att_constraint == 'Y':
            for i in range(group_num):
                tmp = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
                model_list.append(tmp)
                path = "staged_models/model_stage2_Group"+str(group_num)+"-" + str(i)+".pth" 
                model_list[i+1].load_state_dict(torch.load(path,map_location = device))
                model_list[i+1].dnn = Identity()
                model_list[i+1].dnn_linear = Identity()

        history = model.fit(x, y, batch_size = 512, epochs=10, verbose=2, validation_split=0.,model_list = model_list)
        if(add_att_constraint == 'Y'):
            store_path = "model3-Group"+str(group_num)+".pth"
        else:
            store_path = "model3-Group"+str(group_num)+"no_constraint.pth"
        torch.save(model.state_dict(),store_path )
    if flag_stage == 4:
        x, y, feature_columns, behavior_feature_list,all_mid,id_to_histmid_list = get_xy_fd(flag_stage,group_num=group_num)
        model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
        #feature_dict = {'user_id':uid_s1,'hist_mid':hist_mid_s1,'seq_length':seq_length_s1,'mid':mid_s1}
        feature_index = ["user_id","hist_mid","seq_length","mid"]
        if(add_att_constraint == 'Y'):
            path = "model3-Group"+str(group_num)+".pth"
        else:
            path = "model3-Group"+str(group_num)+"no_constraint.pth"
        model.load_state_dict(torch.load(path))
        f2 = open('staged_models/id_to_group{}.pkl'.format(group_num),'rb')
        id_group_index = pickle.load(f2)
        f2.close()
        model.compile('adagrad','binary_crossentropy', metrics=['acc',])
        # this is part of the function evaluate
        pred_ans = model.predict(x, batch_size = 256)
        eval_result = {}
        #for name, metric_fun in model.metrics.items():
        #   eval_result[name] = metric_fun(y, pred_ans)
        # eval_result["precison"] = precision_score(y,np.where(np.array(pred_ans)>0.5,1,0))
        # eval_result["recall"] = recall_score(y,np.where(np.array(pred_ans)>0.5, 1, 0))
        #eval_result["hitrate"] = hitrate_score(y,pred_ans)


        # pred_by_group = [[]]*group_num     # arange the prediction by group
        # y_by_group = [[]]*group_num
        # for i in range(len(x['user_id'])):
        #     index = id_group_index[x['user_id'][i]]
        #     pred_by_group[index] = pred_by_group[index] + [pred_ans[i]]
        #     y_by_group[index] = y_by_group[index] + [y[i]]

        # auc_res = []
        # precision_res = []
        # recall_res = []
        # hit_res = []
       # for i in range(group_num):
        #    tmp_auc = metric_fun(y_by_group[i], pred_by_group[i])
            #tmp_precision = precision_score(y_by_group[i],np.where(np.array(pred_by_group[i])>0.5,1,0))
            #tmp_recall = recall_score(y_by_group[i], np.where(np.array(pred_by_group[i])>0.5,1,0))
            # tmp_hitrate = hitrate(y_by_group[i], pred_by_group[i])
            #auc_res.append(tmp_auc)
            #precision_res.append(tmp_precision)
            #recall_res.append(tmp_recall)
            # hitrate_res.append(tmp_hitrate)
        #auc_group_variance = np.var(auc_res)
        #precision_group_variance = np.var(precision_res)
        #recall_group_variance = np.var(recall_res)
        #hitrate_group_variance = np.var(hitrate_res)
       # epoch_logs = {}
       # eval_str = ""
    ######################Print eva###################
        # for name, result in eval_result.items():
        #     epoch_logs["val_" + name] = round(result,4)
        # for name in model.metrics:
        #     eval_str += " - " + "val_" + name + \
        #                 ": {0: .4f}".format(epoch_logs["val_" + name])
        # eval_str +=" - " + "acc_group_variance: {0: .4f}".format(auc_group_variance) 
        #precision_res = [round(i,4) for i in precision_res]
        #auc_res = [round(i,4) for i in auc_res]
        # recall_res = [round(i,4) for i in recall_res]
        #hitrate_res = [round(i,4) for i in hitrate_res]
        print("================overall metrics=============")

        precision_count = []
        recall_count = []
        number_each_group = []
        hit = 0
        for i in range(group_num):#id_group_index是从0-9
            precision_count.append(0)
            recall_count.append(0)
            number_each_group.append(0)
        # if isinstance(x, dict):
        #     x = [x[feature] for feature in feature_index]
        # for i in range(len(x)):
        #     if len(x[i].shape) == 1:
        #         x[i] = np.expand_dims(x[i], axis=1)  
        # x = torch.from_numpy(np.concatenate(x, axis=-1))  #将输入转化成矩阵,便于后面取出某一个user的data
        # x[i]  第i个user(从0计数)
        # x[i][0] user_id ; x[i][1:50]:hist_mid; x[i][51]: seq_length; x[i][-1]:traget_mid
        auc_dict = {'pred':[],'actual':[]}
        auc_res = []
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
            this_histmid.add(x['mid'][i-1].item())  # 把当前预测的一个mid加进去
            not_click_mid = all_mid - this_histmid
            sample_list = random.sample(list(not_click_mid),100)
            x_thisuser['mid'][1:] = sample_list
            
            #储存sample_list的结果,101个,第一个是真实值
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
        if(add_att_constraint=='Y'):
            print("With Constraint:")
        else:
            print("Without Constraint:")
        print("===============================recall============================")
        print("{0} Groups recall result:{1}".format(group_num,recall_res))
        print("recall variance:{0}".format(round(np.var(recall_res),10)))
        print("===============================precision============================")
        print("{0} Groups precision result:{1}".format(group_num,precision_res))
        print("precision variance:{0}".format(round(np.var(precision_res),10)))
        print("===============================auc============================")
        print("{0} Groups auc result:{1}".format(group_num,auc_res))
        print("precision variance:{0}".format(round(np.var(auc_res),10)))
        import pdb
        pdb.set_trace()
             




                
                
            


        
        #print("=================recall==================")
       # print("auc_res:{0} - auc_group_var:{1:.4f}".format(auc_res,auc_group_variance))
        #print("=================recall==================")
        #print("recall_res:{0} - recall_group_var:{1:.4f}".format(recall_res,recall_group_variance))
        #print("===============precision=================")
        #print("precision_res:{0} - precision_group_var:{1:.4f}".format(precision_res,precision_group_variance))
            #print("===============hitrate=================")
            #print("hitrate_res:{0} - hitrate_group_var:{1}".format(hitrate_res,hitrate_group_variance))
        
     
########################calculate the variance among 10 groups