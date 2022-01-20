import sys, os

sys.path.insert(0, '..')

import numpy as np
import pymysql
import datetime
import torch, random
import os, psutil
from collections import OrderedDict
from tqdm import tqdm
import pickle, json, h5py
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models.din import DIN
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from config import C, print_config_oneline, config_update, argparse, config_oneline, bot
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_xy_fd(flag_stage=1, group_index=0, group_num=0):
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
        # stage_indices[-1] = 0
        mid_s1 = f['mid'][()][stage_indices]
        uid_s1 = f['uid'][()][stage_indices]
        hist_mid_s1 = f['hist_mid'][()][stage_indices]
        import pdb
        seq_length_s1 = f['seq_length'][()][stage_indices]
        rating_s1 = f['rating'][()][stage_indices]
        attrs_s1 = f['attrs'][()][stage_indices]

        f.close()

        import pdb
        if (C.data in ['ali']):
            pass
        else:
            if (flag_stage != 4):
                mid_s1 = mid_s1[:, :C.neg_num + 1].reshape([-1])
                def repeat_func(e):
                    return np.repeat(np.expand_dims(e, axis=-1), C.neg_num + 1, axis=-1)

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
        if (flag_stage != 4):
            return x, y, feature_columns, behavior_feature_list

        all_mid = set()
        id_to_histmid_list = ["padding"]  # 下标从1开始
        for i in range(1, len(uid_s1) + 1):
            thisuser_histmid = set(hist_mid_s1[i - 1])  # 不用把0去掉
            id_to_histmid_list.append(thisuser_histmid)
            all_mid |= thisuser_histmid
        import pdb
        return x, y, feature_columns, behavior_feature_list, all_mid, id_to_histmid_list
        # return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":

    argparse()

    group_list = ["padding"]  # 储存10个分组的id编号
    model_list = ["padding"]  # 储存10个阶段的模型
    id_group_dict = dict()
    add_att_constraint = True
    # x, y, feature_columns, behavior_feature_list = get_xy_fd(flag_stage)
    device = 'cpu'
    use_cuda = True
    gpu_id = C.gpu_id = int(C.gpu_id)
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    # Tmp = input("Press any key to start")
    flag_stage = C.flag_stage = int(C.flag_stage)
    group_num = C.group_num = int(C.group_num)
    assert group_num in [10, 100]
    run = C.run = int(C.run)
    add_att_constraint = 'Y' if C.attReg else 'N'
    print("This training is on Stage" + str(flag_stage))

    # C.batch_size = 256 if C.flag_stage < 4 else 64

    config_update()
    print_config_oneline()
    if (flag_stage == 2):
        print("There are {0} Groups in total".format(group_num))


        def get_all_xy(flag_stage, group_num):
            fname=['hist_mid','mid','seq_length','user_id']

            x_all = []
            y_all = []
            for i in range(0, group_num):
                x, y, _, _ = get_xy_fd(flag_stage, i,group_num)
                x_all.append(x)
                y_all.append(y)
            import pdb
            X = dict()
            for name in fname:
                X[name] = np.concatenate([x_all[i][name] for i in range(0,group_num)],axis=0)
            y = np.concatenate(y_all, axis=0)
            return X,y


        for i in range(0, group_num):
            print("Training on Stage2-Group" + str(i))
            x, y, feature_columns, behavior_feature_list = get_xy_fd(flag_stage, i, group_num)
            import pdb

            seed = random.randint(0, 1000000)
            model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True,
                        seed=seed)
            model.compile('adagrad', 'binary_crossentropy',
                          metrics=['acc'])
            path = C.saved_model_path
            model.load_state_dict(torch.load(path))
            dict_id = list(set(x['user_id']))
            # dict_group = [i] * len(dict_id)
            dict_group = [i for _ in range(len(dict_id))]
            id_group_dict.update(dict(zip(dict_id, dict_group)))
            import pdb

            X, Y = get_all_xy(flag_stage, group_num)  # 用第二阶段所有的数据训练
            history = model.fit(X, Y, batch_size=C.batch_size, epochs=10, verbose=2, validation_split=0)
            path = "staged_models/model_stage2_Group" + str(group_num) + "-" + str(i) + "_{}.pth".format(C.data)
            if not os.path.exists('./staged_models'):
                os.mkdir('./staged_models')
            torch.save(model.state_dict(), path)
            print("Stage: " + str(flag_stage) + "   Stored in : " + path)
            print("\n")
        # import pdb
        ################################################store the id_to_group_dictionary
        f1 = open('staged_models/id_to_group{}_{}.pkl'.format(group_num, C.data), 'wb')
        pickle.dump(id_group_dict, f1)
        f1.close()

    if flag_stage == 1:
        x, y, feature_columns, behavior_feature_list = get_xy_fd(flag_stage, group_num=group_num)
        model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True, seed=C.seed)
        model.compile('adagrad', 'binary_crossentropy',
                      metrics=['auc'])
        history = model.fit(x, y, batch_size=C.batch_size, epochs=C.epoch, verbose=2, validation_split=0)
        # if(group_num==100):
        #     path = "staged_models/model_100groups.pth"
        # if(group_num == 10):
        #     path = "staged_models/model_10groups.pth"
        path = C.model_path
        # path = "staged_models/model_100groups.pth"
        torch.save(model.state_dict(), path)
        print("Stage: " + str(flag_stage) + "   Stored in : " + path)
        print("\n")
    if flag_stage == 3:
        import pdb

        x, y, feature_columns, behavior_feature_list = get_xy_fd(flag_stage, group_num=group_num)
        # import sys
        # import pdb
        # for i, e in enumerate([x, y, feature_columns, behavior_feature_list]):
        #     print("{}, {}".format(i, sys.getsizeof(e)))

        model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True, seed=C.seed)
        model.compile('adagrad', 'binary_crossentropy', metrics=['acc'])
        # load 10 models in stage2
        import pdb

        if C.attReg:
            for i in range(group_num):
                tmp = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True,
                          seed=C.seed, teacher=True)
                model_list.append(tmp)
                path = "staged_models/model_stage2_Group" + str(group_num) + "-" + str(i) + "_{}.pth".format(C.data)
                model_list[i + 1].load_state_dict(torch.load(path, map_location=device))
                # if C.weight_only:
                #     model_list[i + 1].dnn = Identity()
                #     model_list[i + 1].dnn_linear = Identity()

        import pdb
        history = model.fit(x, y, batch_size=C.batch_size, epochs=C.epoch, verbose=2, validation_split=0.,
                            model_list=model_list,flag_stage=3)
        store_path = C.model_path
        torch.save(model.state_dict(), store_path)
        config_save = {'config': config_oneline()}

        # results_save = {'group_auc': auc_res, 'global_auc': eval_result["auc"], "config": config_oneline()}
        with open(C.config_path, 'w') as f:
            json.dump(config_save, f)

    if flag_stage == 4:


        id2group_path = 'staged_models/id_to_group{}_{}.pkl'.format(C.group_num, C.data)
        if not os.path.exists(id2group_path):
            print("There are {0} Groups in total".format(group_num))
            
            for i in range(0,group_num):
                print("Group"+str(i))
                x, y, feature_columns, behavior_feature_list = get_xy_fd(2, i, group_num) 
                import pdb
                dict_id = list(set(x['user_id']))
                # dict_group = [i] * len(dict_id)
                dict_group = [i for _ in range(len(dict_id))]
                id_group_dict.update(dict(zip(dict_id,dict_group)))
                import pdb
            id_group_index = id_group_dict
            with open(id2group_path, 'wb') as f:
                pickle.dump(id_group_dict, f)
        else:
            f2 = open('staged_models/id_to_group{}_{}.pkl'.format(C.group_num, C.data), 'rb')
            id_group_index = pickle.load(f2)


        x, y, feature_columns, behavior_feature_list, all_mid, id_to_histmid_list  = get_xy_fd(flag_stage,group_num=group_num)



        model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True, seed=C.seed)
        path = C.saved_model_path
        model.load_state_dict(torch.load(path))
        # f2 = open('staged_models/id_to_group{}_{}.pkl'.format(C.group_num, C.data), 'rb')
        # id_group_index = pickle.load(f2)
        # import pdb

        # f2.close()
        model.compile('adagrad', 'binary_crossentropy', metrics=['acc', ])
        model.eval()
        # this is part of the function evaluate
        pred_ans = model.predict(x, batch_size=C.batch_size)
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
        index_id = 1
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
        # auc_group_stdiance = np.std(auc_res)
        # # precision_group_stdiance = np.std(precision_res)
        # # recall_group_stdiance = np.std(recall_res)
        # # hitrate_group_stdiance = np.std(hitrate_res)
        # epoch_logs = {}
        # eval_str = ""
        ##### ##################Print eva###################
        # for name, result in eval_result.items():
        #     epoch_logs["val_" + name] = round(result,4)
        # for name in model.metrics:
        #     eval_str += " - " + "val_" + name + \
        #                 ": {0: .4f}".format(epoch_logs["val_" + name])
        # eval_str +=" - " + "acc_group_stdiance: {0: .4f}%".format(auc_group_stdiance * 100)
        # # precision_res = [round(i,4) for i in precision_res]
        # auc_res = [round(i,4) for i in auc_res]
        # # recall_res = [round(i,4) for i in recall_res]
        # # hitrate_res = [round(i,4) for i in hitrate_res]
        # print(eval_str)
        # print(auc_res)
        # print(np.average(auc_res))
        # print("================overall metrics=============")

        # eval_result = {}

        pred_by_group = [[] for _ in range(group_num)]  # arange the prediction by group
        print("================overall metrics=============")

        precision_count = []
        recall5 = []
        recall10 = []
        NDCG5 = []
        NDCG10 = []
        number_each_group = []
        hit = 0
        for i in range(group_num):  # id_group_index是从0-9
            precision_count.append(0)
            recall5.append(0)
            recall10.append(0)
            NDCG5.append(0)
            NDCG10.append(0)
            number_each_group.append(0)

        auc_dict = {'pred': [], 'actual': []}
        auc_res = []
        # 用户真实标签
        one_user_acutalres = [1]
        for i in range(100):
            one_user_acutalres.append(0)
        for i in range(group_num):
            auc_dict['pred'].append([])
            auc_dict['actual'].append([])
            auc_res.append(0)
        x_40users = dict()
        ii_40 = []
        x_40users['user_id'] = np.array([1])
        x_40users['mid'] = np.array([1])
        x_40users['hist_mid'] = np.array([[0 for _ in range(51)]])
        x_40users['seq_length'] = np.array([1])
        x_40users['attrs'] = np.array([[0 for _ in range(C.attr_dim)]])
        import pdb

        for i in tqdm(range(1, len(y) + 1)):
            x_thisuser = dict()
            # predict函数没有将顺序打乱
            x_thisuser['user_id'] = np.array(x['user_id'][i - 1]).repeat(101, axis=0)
            x_thisuser['mid'] = np.array([x['mid'][i - 1]]).repeat(101, axis=0)
            x_thisuser['hist_mid'] = np.array([x['hist_mid'][i - 1]]).repeat(101, axis=0)
            x_thisuser['seq_length'] = np.array(x['seq_length'][i - 1]).repeat(101, axis=0)
            x_thisuser['attrs'] = np.array([x['attrs'][i - 1]]).repeat(101, axis=0)
            this_histmid = id_to_histmid_list[i]
            import pdb

            this_histmid.add(x['mid'][i - 1].item())  # 把当前预测的一个mid加进去
            not_click_mid = all_mid - this_histmid
            sample_list = random.sample(list(not_click_mid), 100)
            x_thisuser['mid'][1:] = sample_list
            # 储存sample_list的结果,101个,第一个是真实值
            import pdb

            ii_40.append(x['user_id'][i - 1])
            if len(ii_40) % 40 == 1:
                x_40users['hist_mid'] = x_thisuser['hist_mid']
                x_40users['user_id'] = x_thisuser['user_id']
                x_40users['mid'] = x_thisuser['mid']
                x_40users['seq_length'] = x_thisuser['seq_length']
                x_40users['attrs'] = x_thisuser['attrs']
            else:
                x_40users['hist_mid'] = np.concatenate((x_40users['hist_mid'], x_thisuser['hist_mid']), axis=0)
                x_40users['user_id'] = np.concatenate((x_40users['user_id'], x_thisuser['user_id']), axis=0)
                x_40users['mid'] = np.concatenate((x_40users['mid'], x_thisuser['mid']), axis=0)
                x_40users['seq_length'] = np.concatenate((x_40users['seq_length'], x_thisuser['seq_length']),
                                                         axis=0)
                import pdb
                x_40users['attrs'] = np.concatenate((x_40users['attrs'], x_thisuser['attrs']),
                                                         axis=0)
            if len(ii_40) == 40 or i == len(y):
                import pdb

                pred_list_40users = list(model.predict(x_40users, batch_size=4096))  # 真实点过的mid预测分数
                # pred_list: (101*40,)
                for j in range(0, len(ii_40)):
                    pred_list = pred_list_40users[j * 101:(j + 1) * 101]
                    auc_dict['pred'][id_group_index.get(ii_40[j], group_num - 1)].extend(pred_list)
                    pred_actualmid = pred_list[0]
                    pred_list.sort(reverse=True)
                    rank = pred_list.index(pred_actualmid) + 1
                    if rank <= 5:
                        precision_count[id_group_index.get(ii_40[j], group_num - 1)] += 1
                        hit += 1
                        recall5[id_group_index.get(ii_40[j], group_num - 1)] += 1
                        NDCG5[id_group_index.get(ii_40[j], group_num - 1)] += (1 / np.log2(rank + 2)) / (
                                1 / np.log2(1 + 2))

                    if rank <= 10:
                        recall10[id_group_index.get(ii_40[j], group_num - 1)] += 1
                        NDCG10[id_group_index.get(ii_40[j], group_num - 1)] += (1 / np.log2(rank + 2)) / (
                                1 / np.log2(1 + 2))
                    number_each_group[id_group_index.get(ii_40[j], group_num - 1)] += 1
                ii_40 = []

            else:
                pass

        import pdb


        recall5_res = [round(recall5[i] / number_each_group[i], 4) for i in range(len(recall5))]
        recall10_res = [round(recall10[i] / number_each_group[i], 4) for i in range(len(recall10))]
        NDCG5_res = [round(NDCG5[i] / number_each_group[i], 4) for i in range(len(NDCG5))]
        NDCG10_res = [round(NDCG10[i] / number_each_group[i], 4) for i in range(len(NDCG10))]
        precision_res = [round(precision_count[i] / 10 / number_each_group[i], 4) for i in range(len(precision_count))]

        for i in range(group_num):
            for j in range(number_each_group[i]):
                auc_dict['actual'][i].extend(one_user_acutalres)  # test的actual数据
            auc_res[i] = roc_auc_score(auc_dict['actual'][i], auc_dict['pred'][i])
        if (add_att_constraint == 'Y'):
            print("With Constraint:")
        else:
            print("Without Constraint:")
        import pdb

        # print("===============================recall============================")
        # print("{0} Groups recall result:{1}".format(group_num,recall_res))
        # print("recall stdiance:{0}".format(round(np.std(recall_res),10)))
        # print("===============================precision============================")
        # print("{0} Groups precision result:{1}".format(group_num,precision_res))
        # print("precision stdiance:{0}".format(round(np.std(precision_res),10)))
        print("===============================auc============================")
        print("{0} Groups auc result:{1}".format(group_num, [round(i, 4) for i in auc_res]))
        print("{0} Groups recall5 result:{1}".format(group_num, [round(i, 4) for i in recall5_res]))
        print("{0} Groups recall10 result:{1}".format(group_num, [round(i, 4) for i in recall10_res]))
        print("{0} Groups NDCG5 result:{1}".format(group_num, [round(i, 4) for i in NDCG5_res]))
        print("{0} Groups NDCG10 result:{1}".format(group_num, [round(i, 4) for i in NDCG10_res]))
        print("AUC      std:{0}".format(round(np.std(auc_res), 10)))
        print("recall5  std:{0}".format(round(np.std(recall5_res), 10)))
        print("recall10 std:{0}".format(round(np.std(recall10_res), 10)))
        print("NDCG5    std:{0}".format(round(np.std(NDCG5_res), 10)))
        print("NDCG10   std:{0}".format(round(np.std(NDCG10_res), 10)))

        import pdb

        results_save = {'group_recall5': recall5_res, 'group_recall10': recall10_res,
                        'group_NDCG5': NDCG5_res, 'group_NDCG10': NDCG10_res, 'group_precision': precision_res,
                        'group_auc': auc_res,
                        'config': config_oneline(), 'number_each_group': number_each_group}

        avg_auc = np.average([np.round(float(e), 5) for e in results_save['group_auc']])
        avg_recall5 = np.average([np.round(float(e), 5) for e in results_save['group_recall5']])
        avg_recall10 = np.average([np.round(float(e), 5) for e in results_save['group_recall10']])
        avg_NDCG5 = np.average([np.round(float(e), 5) for e in results_save['group_NDCG5']])
        avg_NDCG10 = np.average([np.round(float(e), 5) for e in results_save['group_NDCG10']])
        avg_auc = np.round(avg_auc, 4)
        avg_recall5 = np.round(avg_recall5, 4)
        avg_recall10 = np.round(avg_recall10, 4)
        avg_NDCG5 = np.round(avg_NDCG5, 4)
        avg_NDCG10 = np.round(avg_NDCG10, 4)

        print('AVG_AUC:      {}'.format(avg_auc))
        print('AVG_recall5:  {}'.format(avg_recall5))
        print('AVG_recall10: {}'.format(avg_recall10))
        print('AVG_NDCG5:    {}'.format(avg_NDCG5))
        print('AVG_NDCG10:   {}'.format(avg_NDCG10))

        # results_save = {'group_auc': auc_res, 'global_auc': eval_result["auc"], "config": config_oneline()}
        with open(C.result_filepath, 'w') as f:
            json.dump(results_save, f)
        auc_res = [round(i, 4) for i in auc_res]
        recall5_res = [round(i, 4) for i in recall5_res]
        recall10_res = [round(i, 4) for i in recall10_res]
        NDCG5_res = [round(i, 4) for i in NDCG5_res]
        NDCG10_res = [round(i, 4) for i in NDCG10_res]
        auc_std = round(np.std(auc_res), 5)
        recall5_std = round(np.std(recall5_res), 5)
        recall10_std = round(np.std(recall10_res), 5)
        NDCG5_std = round(np.std(NDCG5_res), 5)
        NDCG10_std = round(np.std(NDCG10_res), 5)
        msg = "有一个实验跑完啦!\n" \
              "run_id:{}\n" \
 \
              "auc:{}\n" \
              "avg_auc:{}\n" \
              "auc_std:{}\n" \
 \
              "recall5:{}\n" \
              "avg_recall5:{}\n" \
              "recall5_std:{}\n" \
 \
              "recall10:{}\n" \
              "avg_recall10:{}\n" \
              "recall10_std:{}\n" \
 \
              "NDCG5:{}\n" \
              "avg_NDCG5:{}\n" \
              "NDCG5_std:{}\n" \
 \
              "NDCG10:{}\n" \
              "avg_NDCG10:{}\n" \
              "NDCG10_std:{}\n" \
            .format(C.run,
                    auc_res, avg_auc, auc_std,
                    recall5_res, avg_recall5, recall5_std,
                    recall10_res, avg_recall10, recall10_std,
                    NDCG5_res, avg_NDCG5, NDCG5_std,
                    NDCG10_res, avg_NDCG10, NDCG10_std)
        bot.send_text(msg=msg)
        ############################ Insert the result into Mysql ########################
        connection = pymysql.connect(
            host='sh-cynosdbmysql-grp-d2ld76ic.sql.tencentcdb.com',
            user='deviceCloud',
            port=21240,
            password='L-Y47!hB!z9.zGA',
            db='deviceCloudCD')
        cursor = connection.cursor()
        run_time = str(datetime.date.today())

        sql_base = 'insert into RESULT(run_id,run_time,is_between,exp,cd_lambda,res,avg,std,data,metrics) values({},\'{}\',\'{}\',{},{},\'{}\',{},{},\'{}\',\'{}\')'
        # sql_auc = 'insert into RESULT(run_id,run_time,is_between,fd_lambda,bd_lambda,res,avg,std,data,metrics) values({},\'{}\',\'{}\',{},{},\'{}\',{},{},\'{}\',\'\')' \
        cdl = C.causalD_lambda
        exp = C.exp
        sql_auc = sql_base.format(C.run, run_time, C.CausalD, exp, cdl, str(auc_res), avg_auc, auc_std, C.data, 'auc')
        sql_recall5 = sql_base.format(C.run, run_time, C.CausalD, exp, cdl, str(recall5_res), avg_recall5, recall5_std, C.data, 'recall5')
        sql_recall10 = sql_base.format(C.run, run_time, C.CausalD, exp, cdl, str(recall10_res), avg_recall10, recall10_std, C.data, 'recall10')
        sql_NDCG5 = sql_base.format(C.run, run_time, C.CausalD, exp, cdl, str(NDCG5_res), avg_NDCG5, NDCG5_std, C.data, 'NDCG5')
        sql_NDCG10 = sql_base.format(C.run, run_time, C.CausalD, exp, cdl, str(NDCG10_res), avg_NDCG10, NDCG10_std, C.data, 'NDCG10')
        # sql_recall5 = 'insert into RESULT(run_id,run_time,is_between,fd_lambda,bd_lambda,res,avg,std,data,metrics) values({},\'{}\',\'{}\',{},{},\'{}\',{},{},\'{}\',\'recall5\')' \
        #     .format(C.run, run_time, C.CausalD, C.fd_lambda, C.bd_lambda, str(recall5_res), avg_recall5, recall5_std, C.data)
        # sql_recall10 = 'insert into RESULT(run_id,run_time,is_between,fd_lambda,bd_lambda,res,avg,std,data,metrics) values({},\'{}\',\'{}\',{},{},\'{}\',{},{},\'{}\',\'recall10\')' \
        #     .format(C.run, run_time, C.CausalD, C.fd_lambda, C.bd_lambda, str(recall10_res), avg_recall10, recall10_std, C.data)
        # sql_NDCG5 = 'insert into RESULT(run_id,run_time,is_between,fd_lambda,bd_lambda,res,avg,std,data,metrics) values({},\'{}\',\'{}\',{},{},\'{}\',{},{},\'{}\',\'NDCG5\')' \
        #     .format(C.run, run_time, C.CausalD, C.fd_lambda, C.bd_lambda, str(NDCG5_res), avg_NDCG5, NDCG5_std, C.data)
        # sql_NDCG10 = 'insert into RESULT(run_id,run_time,is_between,fd_lambda,bd_lambda,res,avg,std,data,metrics) values({},\'{}\',\'{}\',{},{},\'{}\',{},{},\'{}\',\'NDCG10\')' \
        #     .format(C.run, run_time, C.CausalD, C.fd_lambda, C.bd_lambda, str(NDCG10_res), avg_NDCG10, NDCG10_std, C.data)
        try:
            cursor.execute(sql_auc)
            cursor.execute(sql_recall5)
            cursor.execute(sql_recall10)
            cursor.execute(sql_NDCG5)
            cursor.execute(sql_NDCG10)
            connection.commit()
            print("The Result Inserted Into Database Successfully!")
        except pymysql.Error as e:
            connection.rollback()
            print("ERROR: Unable To Insert The Result Into Database!")
            print(e.args[0], e.args[1])
        connection.close()

    # print("=================recall==================")
# print("auc_res:{0} - auc_group_std:{1:.4f}".format(auc_res,auc_group_stdiance))
# print("=======:==========recall==================")
# print("recall_res:{0} - recall_group_std:{1:.4f}".format(recall_res,recall_group_stdiance))
# print("===============precision=================")
# print("precision_res:{0} - precision_group_std:{1:.4f}".format(precision_res,precision_group_stdiance))
# print("===============hitrate=================")
# print("hitrate_res:{0} - hitrate_group_std:{1}".format(hitrate_res,hitrate_group_stdiance))
print_config_oneline()

########################calculate the stdiance among 10 groups
