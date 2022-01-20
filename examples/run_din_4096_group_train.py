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
        f = h5py.File(C.h5_data_path.replace('data_10groups', 'data_10groups_group'), 'r')
        print(C.h5_data_path.replace('data_10groups', 'data_10groups_group'))
        # num_user = len(set(f['uid'][()]))
        num_user = f['uid'][()].max()
        max_mid = f['mid'][()].max()
        feature_columns = [SparseFeat('user_id', num_user + 1, embedding_dim=8),
                           SparseFeat('mid', max_mid + 1, embedding_dim=8)]
        feature_columns += [VarLenSparseFeat(SparseFeat('hist_mid', max_mid + 1, embedding_dim=8), C.max_pad_length,
                                             length_name="seq_length")]
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
                hist_mid_s1 = repeat_func(hist_mid_s1).transpose(0, 2, 1).reshape([-1, C.max_pad_length])
                seq_length_s1 = repeat_func(seq_length_s1).reshape([-1])
                rating_s1 = repeat_func(rating_s1)
                rating_s1[:, 1:] = 0
                rating_s1 = rating_s1.reshape([-1])
                import pdb
            else:
                mid_s1 = mid_s1[:, 0]
            import pdb

        feature_dict = {'user_id': uid_s1, 'hist_mid': hist_mid_s1, 'seq_length': seq_length_s1, 'mid': mid_s1}
        x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
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
    import pdb
    group_num = C.group_num = int(C.group_num)
    assert group_num in [10, 100]
    run = C.run = int(C.run)
    add_att_constraint = 'Y' if C.attReg else 'N'
    print("This training is on Stage" + str(flag_stage))

    # C.batch_size = 256 if C.flag_stage < 4 else 64

    config_update()
    print_config_oneline()
    assert flag_stage == 2
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
            # path = C.saved_model_path
            # model.load_state_dict(torch.load(path))
            dict_id = list(set(x['user_id']))
            # dict_group = [i] * len(dict_id)
            dict_group = [i for _ in range(len(dict_id))]
            id_group_dict.update(dict(zip(dict_id, dict_group)))
            import pdb

            X, Y = get_all_xy(flag_stage, group_num)  # 用第二阶段所有的数据训练
            history = model.fit(X, Y, batch_size=C.batch_size, epochs=10, verbose=2, validation_split=0)
            path = "staged_models_group/model_stage2_Group" + str(group_num) + "-" + str(i) + "_{}.pth".format(C.data)
            if not os.path.exists('./staged_models_group'):
                os.mkdir('./staged_models_group')
            torch.save(model.state_dict(), path)
            print("Stage: " + str(flag_stage) + "   Stored in : " + path)
            print("\n")
        # import pdb
        ################################################store the id_to_group_dictionary
        f1 = open('staged_models_group/id_to_group{}_{}.pkl'.format(group_num, C.data), 'wb')
        pickle.dump(id_group_dict, f1)
        f1.close()


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
