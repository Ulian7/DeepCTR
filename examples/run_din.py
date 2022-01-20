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
from config import C, print_config_oneline, config_update, argparse, config_oneline,bot
from sklearn.metrics import accuracy_score,precision_score,recall_score, roc_auc_score
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return x



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
    data_path = C.data_path
    import pickle
    # if(group_num==10):
    #     #f2 = open('../ml-1M/temp_sample.pkl','rb')
    #     f2 = open('../ml-1M/temp_all_10groups.pkl','rb')
    # elif(group_num == 100):
    #     f2 = open('../ml-1M/temp_all_100groups.pkl','rb')
    # else:
    #      raise ValueError("Must input a group num either 10 or 100!")
    if C.hdf:
        import pdb
        f = h5py.File(C.h5_data_path, 'r')
        # num_user = len(set(f['uid'][()]))
        num_user = f['uid'][()].max()
        max_mid = f['mid'][()].max()
        feature_columns = [SparseFeat('user_id', num_user+1, embedding_dim=8), SparseFeat('mid', max_mid+1, embedding_dim=8)]
        feature_columns += [DenseFeat('attrs', C.attr_dim)]
        feature_columns += [VarLenSparseFeat(SparseFeat('hist_mid', max_mid+1, embedding_dim=8), C.max_pad_length, length_name="seq_length")]
        behavior_feature_list = ['mid']
        import pdb
        stage_indices = f['stage{}'.format(flag_stage)]
        if flag_stage == 2:
            stage_indices = stage_indices['group{}'.format(group_index)][()]
        else:
            stage_indices = stage_indices[()]

        import pdb
        # TODO
        #stage_indices[-1] = 0
        mid_s1 = f['mid'][()][stage_indices]
        uid_s1 = f['uid'][()][stage_indices]
        hist_mid_s1 = f['hist_mid'][()][stage_indices]
        import pdb
        seq_length_s1 = f['seq_length'][()][stage_indices]
        rating_s1 = f['rating'][()][stage_indices]
        attrs_s1 = f['attrs'][()][stage_indices]

        f.close()
        
        import pdb
        if(C.data in ['ali']):
            pass
        else:
            if(flag_stage!=4):
                mid_s1 = mid_s1[:, :C.neg_num+1].reshape([-1])
                def repeat_func(e):
                    return np.repeat(np.expand_dims(e, axis=-1), C.neg_num+1, axis=-1)
                import pdb
                uid_s1 = repeat_func(uid_s1).reshape([-1])
                hist_mid_s1 = repeat_func(hist_mid_s1).transpose(0,2,1).reshape([-1, C.max_pad_length])
                attrs_s1 = repeat_func(attrs_s1).transpose(0,2,1).reshape([-1, C.attr_dim])
                
                seq_length_s1 = repeat_func(seq_length_s1).reshape([-1])
                rating_s1 = repeat_func(rating_s1)
                rating_s1[:, 1:] = 0
                rating_s1 = rating_s1.reshape([-1])
                import pdb
            else:
                mid_s1 = mid_s1[:, 0]
            import pdb

        feature_dict = {'user_id':uid_s1,'hist_mid':hist_mid_s1,'seq_length':seq_length_s1,'mid':mid_s1, 'attrs': attrs_s1}
        x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
        import pdb
        y = rating_s1
        import pdb
        if(flag_stage!=4):
            return x, y, feature_columns, behavior_feature_list
        
        all_mid = set()
        id_to_histmid_list = ["padding"]  # 下标从1开始
        for  i in range(1,len(uid_s1)+1):
            this_user_histmid = set(hist_mid_s1[i-1])  #不用把0去掉
            id_to_histmid_list.append(this_user_histmid)
            all_mid |=this_user_histmid
        import pdb
        return x, y, feature_columns, behavior_feature_list,all_mid,id_to_histmid_list
            # return x, y, feature_columns, behavior_feature_list



if __name__ == "__main__":

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
    run = C.run = int(C.run)
    add_att_constraint = 'Y' if C.attReg else 'N'
    print("This training is on Stage"+str(flag_stage))
    
    # C.batch_size = 256 if C.flag_stage < 4 else 64

    config_update()
    print_config_oneline()
    if(flag_stage==2):
        print("There are {0} Groups in total".format(group_num))
        
        for i in range(0,group_num):
            print("Training on Stage2-Group"+str(i))
            x, y, feature_columns, behavior_feature_list = get_xy_fd(flag_stage,i,group_num) 
            import pdb
            seed = random.randint(0, 1000000)
            model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True, seed=C.seed)
            model.compile('adagrad', 'binary_crossentropy',
                  metrics=['acc']) 
            path = C.saved_model_path
            model.load_state_dict(torch.load(path))
            dict_id = list(set(x['user_id']))
            # dict_group = [i] * len(dict_id)
            dict_group = [i for _ in range(len(dict_id))]
            id_group_dict.update(dict(zip(dict_id,dict_group)))
            import pdb
            history = model.fit(x, y, batch_size=C.batch_size, epochs=10, verbose=2, validation_split=0) 
            path = "staged_models/model_stage2_Group"+str(group_num)+"-" + str(i)+"_{}.pth".format(C.data)
            if not os.path.exists('./staged_models'):
                os.mkdir('./staged_models')
            torch.save(model.state_dict(), path)
            print("Stage: "+str(flag_stage)+"   Stored in : "+path)
            print("\n")
        # import pdb
################################################store the id_to_group_dictionary
        f1 = open('staged_models/id_to_group{}_{}.pkl'.format(group_num, C.data),'wb')
        pickle.dump(id_group_dict,f1)
        f1.close()

    if flag_stage == 1:
        x, y, feature_columns, behavior_feature_list = get_xy_fd(flag_stage,group_num=group_num)
        model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True, seed=C.seed)
        model.compile('adagrad', 'binary_crossentropy',
                  metrics=['auc'])
        history = model.fit(x, y, batch_size=C.batch_size, epochs=C.epoch, verbose=2, validation_split=0)
        # if(group_num==100):
        #     path = "staged_models/model_100groups.pth"
        # if(group_num == 10):
        #     path = "staged_models/model_10groups.pth"
        path = C.model_path
        #path = "staged_models/model_100groups.pth"
        torch.save(model.state_dict(), path)
        print("Stage: "+str(flag_stage)+"   Stored in : "+path)
        print("\n")
    if flag_stage == 3:
        import pdb
        x, y, feature_columns, behavior_feature_list = get_xy_fd(flag_stage,group_num=group_num)
        # import sys
        # import pdb
        # for i, e in enumerate([x, y, feature_columns, behavior_feature_list]):
        #     print("{}, {}".format(i, sys.getsizeof(e)))

        model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True, seed=C.seed)
        model.compile('adagrad','binary_crossentropy', metrics=['acc'])
        # load 10 models in stage2
        import pdb
        if C.attReg:
            for i in range(group_num):
                tmp = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True, seed=C.seed, teacher=True)
                model_list.append(tmp)
                path = "staged_models/model_stage2_Group"+str(group_num)+"-" + str(i)+"_{}.pth".format(C.data)
                model_list[i+1].load_state_dict(torch.load(path,map_location = device))
                # if C.weight_only:
                #     model_list[i+1].dnn = Identity()
                #     model_list[i+1].dnn_linear = Identity()
        history = model.fit(x, y, batch_size = C.batch_size, epochs=C.epoch, verbose=2, validation_split=0.,model_list = model_list,flag_stage=3)
        store_path = C.model_path
        torch.save(model.state_dict(),store_path )
        config_save = {'config': config_oneline()}
             
        # results_save = {'group_auc': auc_res, 'global_auc': eval_result["auc"], "config": config_oneline()}
        with open(C.config_path, 'w') as f:
            json.dump(config_save, f)


    if flag_stage == 4:
        x, y, feature_columns, behavior_feature_list, all_mid, id_to_histmid_list  = get_xy_fd(flag_stage,group_num=group_num)

        print("There are {0} Groups in total".format(group_num))
        
        for i in range(0,group_num):
            print("Group"+str(i))
            x, y, feature_columns, behavior_feature_list = get_xy_fd(flag_stage,i,group_num) 
            import pdb
            # seed = random.randint(0, 1000000)
            # model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True, seed=C.seed)
            # model.compile('adagrad', 'binary_crossentropy',
            #       metrics=['acc']) 
            # path = C.saved_model_path
            # model.load_state_dict(torch.load(path))
            dict_id = list(set(x['user_id']))
            # dict_group = [i] * len(dict_id)
            dict_group = [i for _ in range(len(dict_id))]
            id_group_dict.update(dict(zip(dict_id,dict_group)))
            import pdb
            # history = model.fit(x, y, batch_size=C.batch_size, epochs=10, verbose=2, validation_split=0) 
            # path = "staged_models/model_stage2_Group"+str(group_num)+"-" + str(i)+"_{}.pth".format(C.data)
            # if not os.path.exists('./staged_models'):
            #     os.mkdir('./staged_models')
            # torch.save(model.state_dict(), path)
            # print("Stage: "+str(flag_stage)+"   Stored in : "+path)
            # print("\n")




        model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True, seed=C.seed)
        path = C.saved_model_path
        model.load_state_dict(torch.load(path))
        id_group_index = id_group_dict
        # f2 = open('staged_models/id_to_group{}_{}.pkl'.format(C.group_num, C.data),'rb')
        # id_group_index = pickle.load(f2)
        # f2.close()
        model.compile('adagrad','binary_crossentropy', metrics=['acc',])
        model.eval()
        # this is part of the function evaluate
        pred_ans = model.predict(x, batch_size = C.batch_size)
        # eval_result = {}
        # for name, metric_fun in model.metrics.items():
        #     eval_result[name] = metric_fun(y, pred_ans)
        # # eval_result["precison"] = precision_score(y,np.where(np.array(pred_ans)>0.5,1,0))
        # # eval_result["recall"] = recall_score(y,np.where(np.array(pred_ans)>0.5, 1, 0))
        # # eval_result["hitrate"] = hitrate_score(y,pred_ans)


        # pred_by_group = [[] for _ in range(group_num)]    # arange the prediction by group
        # y_by_group = [[] for _ in range(group_num)]
        # for i in range(len(x['user_id'])):
        #     index = id_group_index[x['user_id'][i]]
        #     pred_by_group[index] = pred_by_group[index] + [pred_ans[i]]
        #     y_by_group[index] = y_by_group[index] + [y[i]]

        # auc_res = []
        # precision_res = []
        # recall_res = []
        # hit_res = []
        # number_each_group = {}
        # import pdb
        # for i in range(group_num):
        #     tmp_auc = roc_auc_score(y_by_group[i], pred_by_group[i])
        #     # tmp_precision = precision_score(y_by_group[i],np.where(np.array(pred_by_group[i])>0.5,1,0))
        #     # tmp_recall = recall_score(y_by_group[i], np.where(np.array(pred_by_group[i])>0.5,1,0))
        #     # tmp_hitrate = hitrate(y_by_group[i], pred_by_group[i])
        #     auc_res.append(tmp_auc)
        #     # precision_res.append(tmp_precision)
        #     # recall_res.append(tmp_recall)
        #     # hitrate_res.append(tmp_hitrate)
        #     number_each_group[i] = len(y_by_group[i])
        #     
        # # results_save = np.stack([auc_res, precision_res, recall_res])
        # 
        # results_save = {'group_auc': auc_res, 'config': config_oneline(), 'number_each_group': number_each_group}
        # with open(C.result_filepath, 'w') as f:
        #     json.dump(results_save, f)
        # auc_group_variance = np.var(auc_res)
        # # precision_group_variance = np.var(precision_res)
        # # recall_group_variance = np.var(recall_res)
        # # hitrate_group_variance = np.var(hitrate_res)
        # epoch_logs = {}
        # eval_str = ""
    ##### ##################Print eva###################
        # for name, result in eval_result.items():
        #     epoch_logs["val_" + name] = round(result,4)
        # for name in model.metrics:
        #     eval_str += " - " + "val_" + name + \
        #                 ": {0: .4f}".format(epoch_logs["val_" + name])
        # eval_str +=" - " + "acc_group_variance: {0: .4f}%".format(auc_group_variance * 100) 
        # # precision_res = [round(i,4) for i in precision_res]
        # auc_res = [round(i,4) for i in auc_res]
        # # recall_res = [round(i,4) for i in recall_res]
        # # hitrate_res = [round(i,4) for i in hitrate_res]
        # print(eval_str)
        # print(auc_res)
        # print(np.average(auc_res))
        # print("================overall metrics=============")

        # eval_result = {}

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
        # print("===============================recall============================")
        # print("{0} Groups recall result:{1}".format(group_num,recall_res))
        # print("recall variance:{0}".format(round(np.var(recall_res),10)))
        # print("===============================precision============================")
        # print("{0} Groups precision result:{1}".format(group_num,precision_res))
        # print("precision variance:{0}".format(round(np.var(precision_res),10)))
        print("===============================auc============================")
        print("{0} Groups auc result:{1}".format(group_num,[round(i,4) for i in auc_res]))
        print("AUC variance:{0}".format(round(np.var(auc_res),10)))
        import pdb
        results_save = {'group_recall': recall_res, 'group_precision': precision_res, 'group_auc': auc_res, 'config': config_oneline(), 'number_each_group': number_each_group}
             
        avg_auc = np.average([np.round(float(e), 5) for e in results_save['group_auc']])
        print('AVG_AUC: {}'.format(avg_auc))



        # results_save = {'group_auc': auc_res, 'global_auc': eval_result["auc"], "config": config_oneline()}
        with open(C.result_filepath, 'w') as f:
            json.dump(results_save, f)
        auc_res = [round(i,4) for i in auc_res]
        auc_var = round(np.var(auc_res),10)
        msg = "有一个实验跑完啦! \
               run_id:{0}  \
	       auc:{1} \
               auc_avg:{2} \
               auc_var:{3}".format(C.run,auc_res,avg_auc,auc_var)        
        bot.send_text(msg=msg)        
            


        
        #print("=================recall==================")
       # print("auc_res:{0} - auc_group_var:{1:.4f}".format(auc_res,auc_group_variance))
        #print("=================recall==================")
        #print("recall_res:{0} - recall_group_var:{1:.4f}".format(recall_res,recall_group_variance))
        #print("===============precision=================")
        #print("precision_res:{0} - precision_group_var:{1:.4f}".format(precision_res,precision_group_variance))
            #print("===============hitrate=================")
            #print("hitrate_res:{0} - hitrate_group_var:{1}".format(hitrate_res,hitrate_group_variance))
    print_config_oneline()

        
     
########################calculate the variance among 10 groups


