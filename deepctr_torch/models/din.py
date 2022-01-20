# -*- coding:utf-8 -*-
"""
Author:
    Yuef Zhang
Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""

from .basemodel import BaseModel
from ..inputs import *
from ..layers import *
from ..layers.sequence import AttentionSequencePoolingLayer
from config import C
from . import causalD as utils
import torch.nn as nn
import torch
# import .causalD as utils



class DIN(BaseModel):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    self.CD_linear = nn.Linear(temp, 2 * self.z1_dim * self.z2_dim)
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return:  A PyTorch model instance.

    """

    def __init__(self, dnn_feature_columns, history_feature_list, dnn_use_bn=False,
                 dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=False, l2_reg_dnn=0.0,
                 l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001,
                 seed=1024, task='binary', device='cpu', teacher=False,flag_stage=None):
        super(DIN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device)

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.history_feature_list = history_feature_list

        self.history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))

        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)
        att_emb_dim = self._compute_interest_dim()
############################################add attribute:self.att_score########################################
        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=att_emb_dim,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)
        import pdb
        if C.frontdoor and not teacher:
            self.frontdoor_attn = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                           embedding_dim=128,
                                                           att_activation=att_activation,
                                                           return_score=False,
                                                           supports_masking=False,
                                                           weight_normalization=att_weight_normalization)

                         

        temp=self.compute_input_dim(dnn_feature_columns) - C.attr_dim
        import pdb

        self.dnn = DNN(inputs_dim=temp,
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        if C.CausalD:
            self.z1_dim = self.z2_dim = 8
            self.CD_linear = nn.Linear(temp, 2 * self.z1_dim * self.z2_dim)
            self.dag = utils.DagLayer(self.z1_dim, self.z1_dim)
            self.in_dim = self.z1_dim * self.z2_dim
            self.decoder = nn.Sequential(
                    nn.Linear(self.in_dim, self.in_dim // 2),
                    nn.Linear(self.in_dim // 2, 8),
                    )

        elif C.CausalD_wo_dag:
            self.z1_dim = self.z2_dim = 8
            self.CD_linear = nn.Linear(temp, self.z1_dim * self.z2_dim)


        if C.CausalD or C.CausalD_wo_dag:
            self.gender_cls = nn.Sequential(
                nn.Linear(self.z2_dim, self.z2_dim // 2),
                nn.ELU(),
                nn.Linear(self.z2_dim // 2, 2),
                )
            self.age_reg = nn.Sequential(
                nn.Linear(self.z2_dim, self.z2_dim // 2),
                nn.ELU(),
                nn.Linear(self.z2_dim // 2, 1),
                )
            self.occu_cls = nn.Sequential(
                nn.Linear(self.z2_dim, self.z2_dim * 2),
                nn.ELU(),
                nn.Linear(self.z2_dim * 2, 21),
                )
            self.zip_cls = nn.Sequential(
                nn.Linear(self.z2_dim, self.z2_dim * 2),
                nn.ELU(),
                nn.Linear(self.z2_dim * 2, self.z2_dim * 4),
                nn.ELU(),
                nn.Linear(self.z2_dim * 4, 3439),
                )
            self.title_reg = nn.Sequential(
                nn.Linear(self.z2_dim, self.z2_dim * 2),
                nn.ELU(),
                nn.Linear(self.z2_dim * 2, self.z2_dim * 4),
                nn.ELU(),
                nn.Linear(self.z2_dim * 4, 512),
                )
            self.genres_cls = nn.Sequential(
                nn.Linear(self.z2_dim, self.z2_dim * 2),
                nn.ELU(),
                nn.Linear(self.z2_dim * 2, 18),
                )
            self.dnn = DNN(inputs_dim=self.z1_dim * self.z2_dim,
                           hidden_units=dnn_hidden_units,
                           activation=dnn_activation,
                           dropout_rate=dnn_dropout,
                           l2_reg=l2_reg_dnn,
                           use_bn=dnn_use_bn)






        self.to(device)


    def forward(self, X,only_score=False, front_dic=None):
        import pdb
        _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        # sequence pooling part
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          return_feat_list=self.history_feature_list, to_list=True)
        # target_item的embedding矩阵
        keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
                                         return_feat_list=self.history_fc_names, to_list=True)
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                              to_list=True)
        # 此时dnn_input_emb_list是由sparse_features组成的list矩阵
        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.sparse_varlen_feature_columns)

        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                      self.sparse_varlen_feature_columns, self.device)
        # sequence_embed_list 是和hist_mid相同的可变长的序列特征(和hist_mid不同)
        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)                     # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)                       # [B, T, E] : T the number of hist_item

        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)  # [B, 1]
        # keys_length表示hist_item的长度
        import pdb
        hist,att_score,keys = self.attention(query_emb, keys_emb, keys_length)           # [B, 1, E],[B, 1, T]
        if C.frontdoor and front_dic is not None:
            group_att_score = torch.stack(front_dic['group_attn'], dim=1)
            group_out = torch.bmm(group_att_score, keys)
            group_deep_input_emb = torch.cat((deep_input_emb.repeat(1, C.group_num, 1), group_out), dim=-1)
            group_deep_input_emb = group_deep_input_emb.view(-1, group_deep_input_emb.size(-1))
            group_dnn_input = combined_dnn_input([group_deep_input_emb], dense_value_list)
            group_dnn_output = self.dnn(group_dnn_input)
            group_dnn_output = group_dnn_output.reshape(-1, C.group_num, group_dnn_output.size(-1))
            # group_dnn_logit = self.dnn_linear(group_dnn_output)

            # group_y_pred = self.out(group_dnn_logit)

        # deep part
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        fd_pred = None
        dnn_input = combined_dnn_input([deep_input_emb], [])
        import pdb
        

        causalD_loss = 0.
        if C.CausalD:
            
            CD_dnn_input = self.CD_linear(dnn_input)
            lm, lv = utils.gaussian_parameters(CD_dnn_input)
            lm, lv = lm.reshape([-1, self.z1_dim,self.z2_dim]), torch.ones(lm.size()[0], self.z1_dim, self.z2_dim).to(lm.device)

            pm, pv = torch.zeros(lm.size()).to(lm.device), torch.ones(lv.size()).to(lm.device)
            fdim = self.z1_dim * self.z2_dim
            kl_loss = utils.kl_normal(lm.view(-1, fdim), lv.view(-1, fdim), pm.view(-1, fdim), pv.view(-1, fdim))
            kl_loss = torch.sum(kl_loss)


            
            qm, qv = self.dag.calculate_dag(lm, torch.ones(lm.size()[0], self.z1_dim, self.z2_dim).to(lm.device))
            qm, qv = qm.reshape([-1, self.z1_dim,self.z2_dim]), qv

            attr_loss = 0.
            dense_value = dense_value_list[0]
            # 'gender','age','occupation','zip', 'title','genres', 'title_vec', 'genres_vec'
            gender = dense_value[:,0].type(torch.LongTensor).to(qm.device)
            g_logits = self.gender_cls(qm[:,0,:])
            attr_loss += nn.CrossEntropyLoss(reduction='sum')(g_logits, gender)

            age = dense_value[:,1].to(qm.device) / C.max_age
            a_logits = self.age_reg(qm[:,1,:])
            attr_loss += nn.MSELoss(reduction='sum')(a_logits.squeeze(), age.squeeze())

            occu = dense_value[:,2].type(torch.LongTensor).to(qm.device)
            o_logits = self.occu_cls(qm[:,2,:])
            attr_loss += nn.CrossEntropyLoss(reduction='sum')(o_logits, occu)

            zip_c = dense_value[:,3].type(torch.LongTensor).to(qm.device)
            z_logits = self.zip_cls(qm[:,3,:])
            attr_loss += nn.CrossEntropyLoss(reduction='sum')(z_logits, zip_c)
            import pdb

            title_vec = dense_value[:, 4:516]
            title_vec_pred = self.title_reg(qm[:, 4, :])
            attr_loss += nn.MSELoss(reduction='sum')(title_vec_pred.squeeze(), title_vec.squeeze())

            import pdb
            
            genres_vec = dense_value[:, -18:]
            ge_logits = nn.Sigmoid()(self.genres_cls(qm[:, 5, :]))
            attr_loss += nn.BCELoss(reduction='sum')(ge_logits, genres_vec)
                    

            z = utils.conditional_sample_gaussian(qm, qv * C.lambdav)
            z = z.reshape(z.size(0), -1)
            decoded = self.decoder(z)
            decoded_loss = torch.nn.MSELoss(reduction='sum')(decoded.squeeze(), query_emb.squeeze())


            dag_A = self.dag.A
            acyclicity_loss = utils._h_A(dag_A, dag_A.size(0))
            # CausalVAE loss func
            acyclicity_loss = acyclicity_loss * C.acy1_lambda + acyclicity_loss * acyclicity_loss * C.acy2_lambda


            dnn_input = qm.reshape(qm.size(0), -1)
            
            causalD_loss = kl_loss * C.kl_lambda + attr_loss * C.attr_lambda + decoded_loss * C.decoded_lambda + acyclicity_loss
            # causalD_loss = kl_loss + decoded_loss
        elif C.CausalD_wo_dag:
            CD_dnn_input = self.CD_linear(dnn_input)
            qm = CD_dnn_input.reshape([-1, self.z1_dim, self.z2_dim])
            
            attr_loss = 0.
            dense_value = dense_value_list[0]
            # 'gender','age','occupation','zip', 'title','genres', 'title_vec', 'genres_vec'
            gender = dense_value[:,0].type(torch.LongTensor).to(qm.device)
            g_logits = self.gender_cls(qm[:,0,:])
            attr_loss += nn.CrossEntropyLoss(reduction='sum')(g_logits, gender)

            age = dense_value[:,1].to(qm.device) / C.max_age
            a_logits = self.age_reg(qm[:,1,:])
            attr_loss += nn.MSELoss(reduction='sum')(a_logits.squeeze(), age.squeeze())

            occu = dense_value[:,2].type(torch.LongTensor).to(qm.device)
            o_logits = self.occu_cls(qm[:,2,:])
            attr_loss += nn.CrossEntropyLoss(reduction='sum')(o_logits, occu)

            zip_c = dense_value[:,3].type(torch.LongTensor).to(qm.device)
            z_logits = self.zip_cls(qm[:,3,:])
            attr_loss += nn.CrossEntropyLoss(reduction='sum')(z_logits, zip_c)
            import pdb

            title_vec = dense_value[:, 4:516]
            title_vec_pred = self.title_reg(qm[:, 4, :])
            attr_loss += nn.MSELoss(reduction='sum')(title_vec_pred.squeeze(), title_vec.squeeze())

            import pdb
            
            genres_vec = dense_value[:, -18:]
            ge_logits = nn.Sigmoid()(self.genres_cls(qm[:, 5, :]))
            attr_loss += nn.BCELoss(reduction='sum')(ge_logits, genres_vec)

            causalD_loss = attr_loss * C.attr_lambda
            dnn_input = qm.reshape(qm.size(0), -1)







        dnn_output = self.dnn(dnn_input)
        # if(only_score== True):
        #     return hist,att_score, None,dnn_output




        if C.frontdoor and front_dic is not None:
            group_dnn_output = group_dnn_output.mean(0).unsqueeze(0).repeat(dnn_output.size(0), 1, 1)
            keys_length = torch.ones(dnn_output.size(0)).to(dnn_input.device) * 10
            fd_hist, fd_att_score, fd_keys = self.frontdoor_attn(dnn_output.unsqueeze(1), group_dnn_output, keys_length)
            fd_pred = self.out(self.dnn_linear(fd_hist))


        
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)

        return y_pred,att_score, fd_pred,dnn_output, causalD_loss

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim


if __name__ == '__main__':
    pass
