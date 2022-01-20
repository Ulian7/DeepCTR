
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import pdb
import os
import pymysql

C = OmegaConf.create()

# default seed 1024
C.seed = random.randint(0, 1000000)
C.max_pad_length = 50 + 1
C.attr_dim = 4 + 512 + 18
C.lambdav = 0.001
C.max_age = 56

C.CausalD = False
C.CausalD_wo_dag = False
C.exp = -1




assert not (C.CausalD_wo_dag and C.CausalD)


# 
C.causalD_lambda = 1
C.attr_lambda = 0.001 
C.kl_lambda = 0.1
C.decoded_lambda = 0.1 
C.acy1_lambda = 3
C.acy2_lambda = 0.5



C.weight_reg = False
C.REGLOSS_TYPE = OmegaConf.create()
C.REGLOSS_TYPE.per_kl_iv = 1
C.REGLOSS_TYPE.approx_nDCG = 2
C.REGLOSS_TYPE.endd = 3
C.regloss_type = 1

C.LABEL_TYPE = OmegaConf.create()
C.LABEL_TYPE.MultiLabel = 1

C.weight_only = False

# -------------------------------

# ******
C.model_type = 0

C.backdoor = C.frontdoor = False
if C.model_type == 0:
    pass
elif C.model_type == 1:
    C.backdoor = True
elif C.model_type == 2:
    C.frontdoor = True
elif C.model_type == 3:
    C.backdoor = C.frontdoor = True


# C.backdoor = True
C.BD_TYPE = OmegaConf.create()
C.BD_TYPE.mean = 1
C.BD_TYPE.ada_stable = 2
C.BD_TYPE.attn_kl = 3
C.bd_type = 3
C.bd_type = C.bd_type * C.backdoor

# C.frontdoor = False
C.FD_TYPE = OmegaConf.create()
C.FD_TYPE.ISCS = 1
C.fd_type = 1
C.fd_type = C.fd_type * C.frontdoor
C.fd_lambda = 1


from dingtalkchatbot.chatbot import DingtalkChatbot
# WebHook地址

access_token ='719c90a8620d1f2d2f857ab73764994cdce13cc1cfb453c4140b3747e8f0e473'
webhook = f'https://oapi.dingtalk.com/robot/send?access_token={access_token}'
secret = 'SECe9e212838325ee1865a84f51741e1dc88d96cfcfd98b7963706314d4043df58a'  # 可选：创建机器人勾选“加签”选项时使用
bot = DingtalkChatbot(webhook, secret=secret)  # 方式二：勾选“加签”选项时使用（v1.5以上新功能）

if C.bd_type == C.BD_TYPE.attn_kl or C.weight_reg:
    C.weight_only = True


C.bd_lambda = 0.1

import pdb
C.regloss_type = C.regloss_type * 10
if C.backdoor:
    C.regloss_type = C.regloss_type + C.bd_type

C.regloss_type = C.regloss_type * 10
if C.frontdoor:
    C.regloss_type = C.regloss_type + C.fd_type



def print_config_oneline(return_str = False):
    print(str(OmegaConf.to_yaml(C)).replace("\n", ' '))

def config_oneline():
    return str(OmegaConf.to_yaml(C)).replace("\n", ' ')

def time_string():
    import time
    return time.strftime("%m%d", time.localtime())






def config_update():
    attReg_suffix = "_attReg" if C.attReg else ""
    group_suffix = "_group{}".format(C.group_num)
    run_suffix = "_run{}".format(C.run)
    time_suffix = "_{}".format(time_string())
    data_suffix = "_{}".format(C.data)
    stage_suffix = "_s{}".format(C.flag_stage)
    loss_suffix = "_regloss{}".format(C.exp)
    general_name = "{}{}{}{}{}{}".format(group_suffix, data_suffix, run_suffix, attReg_suffix, loss_suffix, time_suffix)
    C.result_filepath = "./results/results{}.log".format(general_name)
    C.model_path = "models/model{}{}.pth".format(stage_suffix, general_name)
    C.config_path = "models/config{}{}.log".format(stage_suffix, general_name)
    if not os.path.exists('./models'):
        os.mkdir('./models')
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if C.flag_stage == 2:
        C.saved_model_path = "models/model_s1{}.pth".format(general_name)
    elif C.flag_stage == 4:
        if C.date_suffix != "None":
            load_name = "{}{}{}{}{}_{}".format(group_suffix, data_suffix, run_suffix, attReg_suffix, loss_suffix, C.date_suffix)
        else:
            load_name = general_name
        C.saved_model_path = "models/model_s3{}.pth".format(load_name)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  

setup_seed(C.seed)


def argparse():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", '--gpu_id', dest='gpu_id', default=0)
    parser.add_argument("-s", '--stage_id', dest='flag_stage', default=4)
    parser.add_argument("-n", '--neg_num', dest='neg_num', default=5)
    parser.add_argument("-e", '--epoch', dest='epoch', default=10)
    parser.add_argument("-rl", '--reg_lambda', dest='reg_lambda', default=1)
    parser.add_argument("-b", '--batch_size', dest='batch_size', default=1024)
    parser.add_argument("-gn", '--group_num', dest='group_num', default=100)
    parser.add_argument("-r", '--run_id', dest='run', default=0)
    parser.add_argument("-d", '--data', dest='data', default='amazonElectron')
    parser.add_argument('--date', dest='date_suffix', default='None')
    parser.add_argument('--reg', dest='attReg', action='store_true')
    parser.add_argument('--pickle', dest='hdf', action='store_false')
    parser.add_argument('--fd', dest='fd_lambda', default=0)
    parser.add_argument('--bd', dest='bd_lambda', default=0)
    parser.add_argument('--cd', dest='causalD_lambda', default=1)
    parser.add_argument('--gi', dest='group_id', default=0)
    parser.add_argument('--exp', dest='exp', default=-1)

    args = parser.parse_args()


    for k, v in vars(args).items():
        
        C[k] = v


    assert C.data in ['ml-1M', 'amazonElectron', 'amazonBooks','ali']
    C.data_path = '../{}/temp_all_{}groups.pkl'.format(C.data, C.group_num)
    C.h5_data_path = '../{}/data_{}groups.hdf5'.format(C.data, C.group_num)
    C.epoch = int(C.epoch)
    C.batch_size = int(C.batch_size)
    C.reg_lambda = float(C.reg_lambda)
    C.flag_stage = int(C.flag_stage)
    C.bd_lambda = float(C.bd_lambda)
    C.fd_lambda = float(C.fd_lambda)
    C.exp = int(C.exp)
    C.causalD_lambda = float(C.causalD_lambda)
    C.group_id = int(C.group_id)
    if C.regloss_type == 2:
        C.reg_lambda = C.reg_lambda / 100.
    elif C.regloss_type == 3:
        C.reg_lambda = C.reg_lambda / 10.


    if C.flag_stage == 4:
        C.batch_size = 64

    import pdb
    if C.flag_stage in [1,2]:
        C.backdoor = C.frontdoor = False


    C.attr_lambda = C.attr_lambda * C.causalD_lambda
    C.kl_lambda = C.kl_lambda * C.causalD_lambda
    C.decoded_lambda = C.decoded_lambda * C.causalD_lambda

    if C.exp == -1:
        C.CausalD = C.CausalD_wo_dag = False
    elif C.exp == 1:
        C.CausalD = True
    elif C.exp == 2:
        C.CausalD_wo_dag = True

    print("exp: {}".format(C.exp))

#
#    if C.data == 'amazonElectron':
#        C.bd_lambda =0.06 
#        C.fd_lambda = 101
#    elif C.data == 'ml-1M':
#        C.bd_lambda = 0.05
#        C.fd_lambda = 0
#    elif C.data == 'ali':
#        C.bd_lambda = 0
#        C.fd_lambda = 0












