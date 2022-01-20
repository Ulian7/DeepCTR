




import pdb
import torch
from omegaconf import OmegaConf

LABEL_TYPE = OmegaConf.create()
LABEL_TYPE.MultiLabel = 1
LABEL_TYPE.Permutation = 2

# from ptranking.data.data_utils import LABEL_TYPE
# from ptranking.base.ranker import NeuralRanker
# from ptranking.ltr_adhoc.eval.parameter import ModelParameter
# from ptranking.metric.adhoc_metric import torch_dcg_at_k
# from ptranking.base.neural_utils import robust_sigmoid
class Robust_Sigmoid(torch.autograd.Function):
    ''' Aiming for a stable sigmoid operator with specified sigma '''

    @staticmethod
    def forward(ctx, input, sigma=1.0, gpu=False):
        '''
        :param ctx:
        :param input: the input tensor
        :param sigma: the scaling constant
        :return:
        '''
        x = input if 1.0==sigma else sigma * input

        torch_half = torch.cuda.FloatTensor([0.5]) if gpu else torch.FloatTensor([0.5])
        sigmoid_x_pos = torch.where(input>0, 1./(1. + torch.exp(-x)), torch_half)

        exp_x = torch.exp(x)
        sigmoid_x = torch.where(input<0, exp_x/(1.+exp_x), sigmoid_x_pos)

        grad = sigmoid_x * (1. - sigmoid_x) if 1.0==sigma else sigma * sigmoid_x * (1. - sigmoid_x)
        ctx.save_for_backward(grad)

        return sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        '''
        :param ctx:
        :param grad_output: backpropagated gradients from upper module(s)
        :return:
        '''
        grad = ctx.saved_tensors[0]

        bg = grad_output * grad # chain rule

        return bg, None, None

#- function: robust_sigmoid-#
robust_sigmoid = Robust_Sigmoid.apply


def torch_dcg_at_k(batch_sorted_labels, cutoff=None, label_type=LABEL_TYPE.MultiLabel, gpu=False):
    '''
    ICML-nDCG, which places stronger emphasis on retrieving relevant documents
    :param batch_sorted_labels: [batch_size, ranking_size] a batch of ranked labels (either standard or predicted by a system)
    :param cutoff: the cutoff position
    :param label_type: either the case of multi-level relevance or the case of listwise int-value, e.g., MQ2007-list
    :return: [batch_size, 1] cumulative gains for each rank position
    '''
    if cutoff is None: # using whole list
        cutoff = batch_sorted_labels.size(1)

    if LABEL_TYPE.MultiLabel == label_type:    #the common case with multi-level labels
        batch_numerators = torch.pow(2.0, batch_sorted_labels[:, 0:cutoff]) - 1.0
    elif LABEL_TYPE.Permutation == label_type: # the case like listwise ltr_adhoc, where the relevance is labeled as (n-rank_position)
        batch_numerators = batch_sorted_labels[:, 0:cutoff]
    else:
        raise NotImplementedError
#     pdb.set_trace()
    batch_discounts = torch.log2(torch.arange(cutoff).type(torch.cuda.FloatTensor).expand_as(batch_numerators) + 2.0) if gpu \
                                            else torch.log2(torch.arange(cutoff).type(torch.cuda.FloatTensor).expand_as(batch_numerators) + 2.0)
    batch_dcg_at_k = torch.sum(batch_numerators/batch_discounts, dim=1, keepdim=True)
    return batch_dcg_at_k


def get_approx_ranks(input, alpha=10, gpu=False):
    ''' get approximated rank positions: Equation-11 in the paper'''
    batch_pred_diffs = torch.unsqueeze(input, dim=2) - torch.unsqueeze(input, dim=1)  # computing pairwise differences, i.e., Sij or Sxy

    batch_indicators = robust_sigmoid(torch.transpose(batch_pred_diffs, dim0=1, dim1=2), alpha, gpu) # using {-1.0*} may lead to a poor performance when compared with the above way;

    batch_hat_pis = torch.sum(batch_indicators, dim=2) + 0.5  # get approximated rank positions, i.e., hat_pi(x)

    return batch_hat_pis


def approxNDCG(batch_preds=None, batch_stds=None, alpha=10, label_type=None, gpu=False):
    batch_hat_pis = get_approx_ranks(batch_preds, alpha=alpha, gpu=gpu)

    ''' since the input standard labels are sorted in advance, thus directly used '''
    # sorted_labels, _ = torch.sort(batch_stds, dim=1, descending=True)  # for optimal ltr_adhoc based on standard labels
    batch_idcgs = torch_dcg_at_k(batch_sorted_labels=batch_stds, cutoff=None, label_type=label_type)  # ideal dcg given standard labels

    batch_gains = torch.pow(2.0, batch_stds) - 1.0

    batch_dcg = torch.sum(torch.div(batch_gains, torch.log2(batch_hat_pis + 1)), dim=1)
    batch_approx_nDCG = torch.div(batch_dcg, batch_idcgs)

    return batch_approx_nDCG


def approxNDCG_loss(batch_preds=None, batch_stds=None, alpha=10, label_type=None, gpu=False, reduction='sum'):
    batch_hat_pis = get_approx_ranks(batch_preds, alpha=alpha, gpu=gpu)

    # ideal dcg given optimally ordered labels
    batch_idcgs = torch_dcg_at_k(batch_sorted_labels=batch_stds, cutoff=None, label_type=label_type, gpu=gpu)

    if LABEL_TYPE.MultiLabel == label_type:
        batch_gains = torch.pow(2.0, batch_stds) - 1.0
    elif LABEL_TYPE.Permutation == label_type:
        batch_gains = batch_stds
    else:
        raise NotImplementedError

    batch_dcg = torch.sum(torch.div(batch_gains, torch.log2(batch_hat_pis + 1)), dim=1)
    batch_approx_nDCG = torch.div(batch_dcg, batch_idcgs)

    reduction_func = torch.sum if reduction == 'sum' else torch.mean
    batch_loss = -reduction_func(batch_approx_nDCG)
    return batch_loss


if __name__ == '__main__':
    batch_stds = torch.tensor([[0.1, 0.2], [0.5, 0.3]])
    batch_preds = torch.tensor([[0.1, 0.2], [0.5, 0.4]])

    target_batch_stds, batch_sorted_inds = torch.sort(batch_stds, dim=1, descending=True)
    target_batch_preds = torch.gather(batch_preds, dim=1, index=batch_sorted_inds)
    batch_loss = approxNDCG_loss(target_batch_preds, target_batch_stds, label_type=LABEL_TYPE.MultiLabel, reduction='sum')
    print(batch_loss)




