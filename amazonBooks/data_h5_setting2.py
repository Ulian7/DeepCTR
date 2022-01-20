import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import h5py
import os 
import random
dir_path = os.path.dirname(os.path.realpath(__file__))
# unames =['user_id','gender','age','occupation','zip']
# users = pd.read_table(dir_path + '/users.dat', sep='::',
#                       header=None, names=unames)
# rnames =['user_id','movie_id','rating','timestamp']
# ratings = pd.read_table(dir_path +'/ratings.dat', sep='::',
#                         header=None, names=rnames)
# mnames =['movie_id','title','genres']
# movies = pd.read_table(dir_path +'/movies.dat', sep='::',
#                        header=None, names=mnames)
# user_num = ratings.value_counts('user_id')
import pdb
import time

# --------------  Frequently Modified --------------------

group_num = 10    # 分成多少组
neg_num = 20
random_group = False
# --------------------------------------------------------


# unames = ['user_id', 'movie_id', 'rating', 'timestamp']
unames = ['movie_id', 'user_id', 'rating', 'timestamp']
data = pd.read_table(dir_path + '/Books.csv', sep=',',
        header=None, names=unames)
data = data.sort_values(by = ['user_id','timestamp'],ascending=[True,True])
data.loc[data['rating']<=3,'rating'] = 1
data.loc[data['rating']>=4,'rating'] = 1  #先后顺序不可以换

data = data[['user_id','movie_id', 'rating', 'timestamp']] # select the features we need


import pdb
K_num = 16
num_count = data['movie_id'].value_counts()
coreK = num_count.loc[num_count >= K_num]
data = data[data['movie_id'].isin(coreK.index)]
data = data.sort_values(by = ['movie_id'])

num_count = data['user_id'].value_counts()
coreK = num_count.loc[num_count >= K_num]
data = data[data['user_id'].isin(coreK.index)]
data = data.sort_values(by = ['user_id'])
data['rating'] = data['rating'].astype('int64')


data_np = data.to_numpy()

uix = 1

import pdb
pre_uid = None
uid_list = []
from tqdm import tqdm
for i in tqdm(range(len(data))):
    if pre_uid is None:
        pre_uid = data_np[i, 0]
    elif data_np[i,0] == pre_uid:
        pass
    else:
        uix += 1
        pre_uid = data_np[i, 0]
    uid_list.append(uix)



user_num = max(uid_list)

import pdb

data['user_id'] = uid_list

import pdb

ix = 1
data = data.sort_values(by = ['movie_id'])
data_np = data.to_numpy()
import pdb
pre_iid = None
iid_list = []
from tqdm import tqdm
for i in tqdm(range(len(data))):
    if pre_iid is None:
        pre_iid = data_np[i,1]
    elif data_np[i,1] == pre_iid:
        pass
    else:
        ix += 1
        pre_iid = data_np[i, 1]
    iid_list.append(ix)

data['movie_id'] = iid_list
item_num = max(iid_list)
    

import pdb


# data = pd.merge(pd.merge(ratings,users),movies)
data = data.sort_values(by = ['user_id','timestamp'],ascending=[True,True])
data.loc[data['rating']<=3,'rating'] = 1
data.loc[data['rating']>=4,'rating'] = 1  #先后顺序不可以换
# data.loc[data['gender']=='F',['gender']] = 0
# data.loc[data['gender']=='M',['gender']] = 1

data = data[['user_id','movie_id', 'rating']] # select the features we need
# 取前 1W
#data = data[:10000]
num_count = data['user_id'].value_counts()
import pdb
pdb.set_trace()



user_sum = len(num_count)
user_sum = len(num_count)   #总共有多少user
num_each_group = user_sum // group_num
id_list = []
count = 0
for i in range(1,user_sum+1):
    one_list = []
    one_list = one_list + [count]
    count = count + num_count[i]
    one_list = one_list + [count]
    id_list.append(one_list)
    # id_list[k] = [i,j],所有10000条数据中,user k的数据为[i,j)
id_to_seq_len_dic = data.user_id.value_counts()
user_num = len(id_to_seq_len_dic)
data_dict = {}
uid=[]
mid=[]
rating=[]
hist_mid=[]
seq_length=[]
one_hist=[]
stage1=[]#
stage3=[]
# stage2=[[]]*group_num
stage2=[[] for _ in range(group_num)]
stage4=[]
data = data.to_numpy()
# data = np.array(data)

#逐个生成每一个train data,从 1 开始标上index,然后加到data_dict中,最后判断这个点是处于哪个阶段,把index加入对应的字典

if random_group == True:
    num_count.index = random.sample(range(1,user_num+1),user_sum )
num_index_dict = dict([num_count.index[i],i+1] for i in  range(len(num_count)))
#num_index_dict[i] = j 序号为i的user的hist_item数量排第j位,便于stage2的分组

for i in tqdm(range(1,user_num+1)):
    base_id = len(uid)    #此前产生的data个数
    one_user = data[id_list[i-1][0]:id_list[i-1][1]]
    #one_user = data[data['user_id']==i]
    hist_len = len(one_user) #历史点击个数
    data_len = hist_len-1   #产生的data个数
    s4_index = hist_len-1             #index == hist_len-1 ==> 属于 s4_index
    s1_num = data_len//3
    s1_index = s1_num         #index <= s1_index
    s2_index = s1_num*2
    pos_list = [one_user[e][1] for e in range(hist_len)]
    def gen_neg(pos_list, item_num):
        neg = random.randint(1, item_num)
        while neg in pos_list:
            neg = random.randint(1, item_num)
        return neg
        
    for index in range(1,hist_len):  #遍历该user的每一条记录,从第1条开始(0,1,2,3...)
        uid.append(one_user[index][0])
        one_mid = [one_user[index][1]]
        for _ in range(neg_num):
            one_mid.append(gen_neg(pos_list, item_num))
        mid.append(one_mid)
        rating.append(one_user[index][-1])
        if index == 1:
            one_hist = [one_user[0][1]]  #第一份数据
        else:
            one_hist = hist_mid[-1]+[one_user[index-1][1]]#movie_id
        hist_mid.append(one_hist)
        #hist_mid = hist_mid + [one_hist]
        seq_length.append(len(one_hist))
        index_in_all = base_id + index -1  
        if index <= s1_index:
            stage1.append(index_in_all)
        
        elif index<=s2_index:   # 如果是加到 stage2
            pos = num_index_dict[i]//num_each_group  #i是用户序号,pos是对应组别
            if pos == 1:
                import pdb
            if pos==group_num:
                pos=group_num-1   # 多余的人放到最后一组,从0计数
            stage2[pos].append(index_in_all)
            stage3.append(index_in_all)  
        elif index<s4_index:
            stage3.append(index_in_all)  
        else:
            stage4.append(index_in_all)
import pdb



import time

# for i in tqdm(range(1,user_num+1)):
#     base_id = len(uid)    #此前产生的data个数
#     time1 = time.time()
#     one_user = data[id_list[i-1][0]:id_list[i-1][1]]
#     time2 = time.time()
#     print("Cost2: {:.4f}".format(time2 - time1))
#     #one_user = data[data['user_id']==i]
#     hist_len = len(one_user) #历史点击个数
#     data_len = hist_len-1   #产生的data个数
#     s4_index = hist_len-1             #index == hist_len-1 ==> 属于 s4_index
#     s1_num = data_len//3
#     s1_index = s1_num         #index <= s1_index
#     s2_index = s1_num*2
#     time3 = time.time()
#     print("Cost3: {:.4f}".format(time3 - time2))
#     import pdb
#     for index in range(1,hist_len):  #遍历该user的每一条记录,从第1条开始(0,1,2,3...)
#         import pdb
#         time4 = time.time()
#         uid.append(one_user[index][0])
#         mid.append(one_user[index][1])
#         rating.append(one_user[index][-1])
#         if index == 1:
#             one_hist = [one_user[0][1]]  #第一份数据
#         else:
#             one_hist = hist_mid[-1]+[one_user[index-1][1]]#movie_id
#         time5 = time.time()
#         print("Cost5: {:.4f}".format(time5 - time4))
#         hist_mid = hist_mid + [one_hist]
#         time6 = time.time()
#         print("Cost6: {:.4f}".format(time6 - time5))
#         seq_length.append(len(one_hist))
#         index_in_all = base_id + index -1  
#         if index <= s1_index:
#             stage1.append(index_in_all)
#         
#         elif index<=s2_index:   # 如果是加到 stage2
#             pos = num_index_dict[i]//num_each_group  #i是用户序号,pos是对应组别
#             if pos==group_num:
#                 pos=group_num-1   # 多余的人放到最后一组,从0计数
#             stage2[pos] = stage2[pos] + [index_in_all]
#         elif index<s4_index:
#             stage3.append(index_in_all)  
#         else:
#             stage4.append(index_in_all)
#         time7 = time.time()
#         print("Cost7: {:.4f}".format(time7 - time6))
# 
#     time8 = time.time()
#     print("Cost8: {:.4f}".format(time8 - time3))


f1 = h5py.File('data_{}groups.hdf5'.format(group_num), 'w')
uid_np = np.array(uid)
uid_hdf = f1.create_dataset("uid", data=uid_np)

mid_np = np.array(mid)
mid_hdf = f1.create_dataset("mid", data=mid_np)

rating_np = np.array(rating)
rating_ndf = f1.create_dataset("rating", data=rating_np)


from tensorflow.python.keras.preprocessing.sequence import pad_sequences
hist_mid_np = pad_sequences(hist_mid,maxlen = 50+1,padding = 'pre')
hist_mid_hdf = f1.create_dataset("hist_mid", data = hist_mid_np)

seq_length_np = np.array(seq_length)
seq_length_hdf = f1.create_dataset('seq_length', data=seq_length_np)

for ei, e in zip([1,3,4], [stage1, stage3, stage4]):
    e_np = np.array(e)
    e_hdf = f1.create_dataset('stage{}'.format(ei), data=e_np)

stage2_grp = f1.create_group('stage2')
for ei, e in enumerate(stage2):
    e_np = np.array(e)
    e_hdf = stage2_grp.create_dataset('group{}'.format(ei), data=e_np)

# f1 = open('temp_all_{}groups.pkl'.format(group_num),'wb')
# 
# 
# 
# 
# pickle.dump(uid,f1)
# pickle.dump(mid,f1)
# pickle.dump(rating,f1)
# pickle.dump(hist_mid,f1)
# pickle.dump(seq_length,f1)
# pickle.dump(stage1,f1)
# pickle.dump(stage2,f1)
# pickle.dump(stage3,f1)
# pickle.dump(stage4,f1)
# f1.close()
        
