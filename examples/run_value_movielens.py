import numpy as np
import pandas as pd
import torch
import pdb
import sys
sys.path.append('../')
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DCNMix,DeepFM,DCN,DIN

import argparse

parser = argparse.ArgumentParser(description = 'input the model name ')
parser.add_argument('model_name',type = str,help = 'The Model We Use')

args = parser.parse_args()
import warnings

warnings.filterwarnings('ignore')


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


if __name__ == "__main__":
    data = pd.read_csv("./modified_sample.txt")
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", ]
    target = ['rating']

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
       # pdb.set_trace()
        data[feat] = lbe.fit_transform(data[feat])  # fit date[feat] into a dictionary in decreasing order and transform it using the dictionary
    # preprocess the sequence feature

    key2index = {}
    genres_list = list(map(split, data['genres'].values))
    genres_length = np.array(list(map(len, genres_list)))
    max_len = max(genres_length)
    # Notice : padding=`post`
    genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )
    # (146,5)

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                              for feat in sparse_features]
   #fixlen_feature_columns是7个SparseFeat,每一个储存一个feature的信息
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
        key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean')]  # Notice : value 0 is for padding for sequence input feature


    linear_feature_columns = fixlen_feature_columns #+ varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns #+ varlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    model_input = {name: data[name] for name in sparse_features}  #
    model_input["genres"] = genres_list
    
#----------------------------------------------------------------
#version:0.1-------gender_mask        
#-------------------------------------------------------

    age_1to18_mask = [1 if (i==0) | (i==1) else 0 for i in model_input['age'] ]
    age_25to35_mask =  [1 if (i==2) | (i==3) else 0 for i in model_input['age'] ]
    age_45to56_mask =  [1 if (i==4) | (i==5) | (i==6) else 0 for i in model_input['age'] ]
    
    age_mask = np.array([age_1to18_mask,age_25to35_mask,age_45to56_mask])
    
    female_mask = [1 if (i==0) else 0 for i in model_input['gender'] ]
    male_mask = [1 if (i==1) else 0 for i in model_input['gender']   ]
    gender_mask = np.array([female_mask,male_mask])
#     pdb.set_trace()

    # 4.Define Model,compile and train

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    if args.model_name == 'DCNMix':
        print('This Training is on The DCNMix Model....')
        model = DCNMix(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
    elif args.model_name == 'DeepFM':
        print('This Training is on the DeepFM Model...')
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
    elif args.model_name == 'DIN':
        # 排序,每一个user按照所看电影前后顺序排序
        data = data.sort_values(by = ['user_id','timestamp'],ascending=[True,True])
        import ipdb
        ipdb.set_trace()
        hist_num = np.array(data.value_counts('user_id'))  # 统计 每位用户观看电影数量
        user_max_len = hist_num[0]
        user_index_list =[]
        uid = []
        ugender = []
        uoccupation = []
        uzip = []
        uage = []
        hist_iid = []
        this_user = []
        behavior_length = []
        end = False
        for i in range(len(data)):  #遍历每一行数据
            if data.iloc[i]['user_id'] not in uid:  #该行数据不在uid中----是新出现的id
                this_user.append(data.iloc[i]['movie_id'])  
                end = True
                #print(data.iloc[i]['user_id'])
                user_index_list.append(i)
                uid.append(data.iloc[i]['user_id'])
                uoccupation.append(data.iloc[i]['occupation'])
                #uzip.append(data.iloc[i]['zip'])
                uage.append(data.iloc[i]['age'])
                ugender.append(data.iloc[i]['gender'])

            else: 
                this_user.append(data.iloc[i]['movie_id'])
                end = False

            if(end): 
                hist_iid.append(this_user)
                behavior_length.append(len(this_user))
                this_user = []
        hist_iid = pad_sequences(hist_iid,maxlen = user_max_len,padding = 'post')
#-------------------construct X and Y as Input-----------------------------------------------
        feature_dict = {'user': uid, 'gender': ugender, 'movie_id': iid, '': igender,
                    'hist_movie_id': hist_iid, 'age':uage,'occupation':uocupation,
                    'zip':uzip,
                    "seq_length": behavior_length}
      
        
        
        DIN_feature_columns = fixlen_feature_columns + varlen_feature_columns
        DIN_feature_columns += [VarLenSparseFeat(SparseFeat('hist_movie_id', data['hist_movie_id'].nunique(), embedding_dim=4), 4,                                length_name="seq_length")]
        DIN_feature_columns += varlen_feature_columns
        DIN_behavior_feature_list = ['movie_id']
        print('This Training is on the DIN Model...')
        
        model = DIN(DIN_feature_columns, DIN_behavior_feature_list, device=device, att_weight_normalization=True)
    elif args.model_name == 'DCN':
        print('This Training is on the DCN Model...')
        model = DCN(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
        

    model.compile("adam", "binary_crossentropy", metrics=['accuracy'], )
    
#     import pdb
#     pdb.set_trace()
#     The ages are represented by 0,...6: 
#       id                real age
#       0                    1
#       1                    18
#       2                    25
#       3                    35
#       4                    45
#       5                    50
#       6                    56
#      x[3] is the age column
    if  args.model_name == 'DIN':
        x = {name: feature_dict[name] for name in get_feature_names(DIN_feature_columns)}
        history = model.fit(model_input, data[target].values, batch_size=256, epochs=10, verbose=2, validation_split=0.2)
    else:
        history = model.fit(model_input, data[target].values, batch_size=256, epochs=10, verbose=2, validation_split=0.2)
    