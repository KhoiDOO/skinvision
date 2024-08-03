from torch import nn
from typing import *
from torchmetrics.classification import BinaryAccuracy
from sklearn.metrics import roc_curve, auc, roc_auc_score

import torch.nn.functional as F
import torch
import numpy as np


class GlobalAvgPooling(nn.Module):
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()

    def forward(self, x):
        return x.mean(dim=(2, 3))


def parse_loss(cfg: Dict)->nn.Module:
    loss = getattr(torch.nn, cfg.name)(**cfg.args)
    return loss

def parse_optimizer(cfg: Dict, model:nn.Module)->torch.optim.Optimizer:
    params = model.parameters()
    optim = getattr(torch.optim, cfg.name)(params, **cfg.args)
    return optim

def parse_scheduler(cfg: Dict, optimizer: torch.optim.Optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.name)(optimizer, **cfg.args)
    return lr_scheduler

def score(logit: torch.Tensor, label: torch.Tensor, min_tpr: float=0.80):
    pred = torch.round(logit)
    
    solution = label.detach().cpu().numpy()
    submission = pred.detach().cpu().numpy()

    v_gt = abs(np.asarray(solution)-1)
    v_pred = -1.0*np.asarray(submission)

    max_fpr = abs(1-min_tpr)

    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
        
    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    if fpr.shape[0] <= 1 or tpr.shape[0] <= 1:
        return 0
    partial_auc = auc(fpr, tpr)

    return partial_auc.item()

def acc(logit: torch.Tensor, label: torch.Tensor):
    B = logit.size(0)
    pred = torch.round(logit)
    acc = (pred == label).sum() / B

    return acc.item()


if __name__ == '__main__':
    logit = torch.tensor([0.8, 0.85, 0.9, 0.01]*8)
    label = torch.tensor([1, 1, 1, 0]*8)

    print(logit.shape, label.shape)

    par_auc = score(logit, label)
    accuracy = acc(logit, label)

    print(par_auc, type(par_auc))
    print(accuracy, type(par_auc))