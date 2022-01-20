
import torch    

tempature = 2.5
ensemble_epsilon = 1e-3
tp_scaling = 1 - ensemble_epsilon
smooth_val = 1e-8



def endd_loss(ensemble_logits, logits):
    '''
    ensemble_logits are the outputs from our ensemble (batch x ensembles x classes)
    logits are the predicted outputs from our model (batch x classes)
    '''
    # logits = tf.cast(logits, dtype=tf.float64)
    # ensemble_logits = tf.cast(ensemble_logits, dtype=tf.float64)
    import pdb
    alphas = torch.exp(logits / tempature)

    precision = torch.sum(alphas, axis=1)  #sum over classes

    ensemble_probs = torch.softmax(ensemble_logits / tempature, axis=2)  #softmax over classes
    # Smooth for num. stability:
    probs_mean = 1 / ensemble_probs.shape[2]  #divide by nr of classes
    # Subtract mean, scale down, add mean back)
    ensemble_probs = tp_scaling * (ensemble_probs - probs_mean) + probs_mean

    log_ensemble_probs_geo_mean = torch.mean(torch.log(ensemble_probs + smooth_val),
                                              axis=1)  #mean over ensembles

    target_independent_term = torch.mean(torch.lgamma(alphas + smooth_val), axis=1) - torch.lgamma(
        precision + smooth_val)  #sum over lgammma of classes - lgamma(precision)

    target_dependent_term = -torch.sum(
        (alphas - 1.) * log_ensemble_probs_geo_mean, axis=1)  # -sum over classes

    cost = target_dependent_term + target_independent_term
    # tf.print(self.temp)
    return torch.mean(cost) * (tempature**2)  #mean of all batches







