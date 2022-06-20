import torch.nn as nn
import torch


class custom_loss(nn.Module):
    def __init__(self, weight, pref_for_one=1.):
        '''
        This is class that implement asymmetric BCE_loss. It is required for imbalance
        data to prevent overfitting and becoming constant classifier.
        :param weight: Weights to compute loss
        :param pref_for_one: how much attention is required to give to errors on class 1.
        '''
        super(custom_loss, self).__init__()
        assert pref_for_one >= 0, 'Attention to class one must be positive!'
        self.weight = weight
        self.one_w = pref_for_one

    def forward(self, pred, true):
        pred = torch.clamp(pred, 1e-5, 1 - 1e-5)
        loss1 = -self.one_w * true * torch.log(pred)
        loss2 = -(1 - true) * torch.log(1. - pred)
        if self.weight is None:
            loss = torch.mean(loss1 + loss2)
        else:
            loss = torch.mean(self.weight * loss1 + (1. - self.weight) * loss2)
        return loss


def tp_metric(pred, target):
    pred = (pred >= 0.5).float()
    return (pred * target).sum(dim=0)


def fp_metric(pred, target):
    pred = (pred >= 0.5).float()
    return (pred * (1. - target)).sum(dim=0)


def fn_metric(pred, target):
    pred = (pred >= 0.5).float()
    return ((1. - pred) * target).sum(dim=0)


def tn_metric(pred, target):
    pred = (pred >= 0.5).float()
    return ((1. - pred) * (1. - target)).sum(dim=0)


def precision_metric(pred, target):
    tp = tp_metric(pred, target)
    fp = fp_metric(pred, target)
    return tp / (tp + fp + 1e-8)


def recall_metric(pred, target):
    tp = tp_metric(pred, target)
    fn = fn_metric(pred, target)
    return tp / (tp + fn + 1e-8)


def specificity_metric(pred, target):
    tn = tn_metric(pred, target)
    fp = fp_metric(pred, target)
    return tn / (tn + fp + 1e-8)


def f1_score(pred, target):
    precision = precision_metric(pred, target)
    recall = recall_metric(pred, target)
    return 2 * (precision * recall) / (precision + recall + 1e-8)


def NPV_metric(pred, target):
    # negative predictive value
    tn = tn_metric(pred, target)
    fn = fn_metric(pred, target)
    return tn / (tn + fn + 1e-8)


def DOR_metric(pred, target):
    # ratio of the odds of being true positive to the odds of being false positive
    tp = tp_metric(pred, target)
    fp = fp_metric(pred, target)
    tn = tn_metric(pred, target) + 1e-8
    fn = fn_metric(pred, target) + 1e-8
    return (tp / fn) / (fp / tn + 1e-8)
