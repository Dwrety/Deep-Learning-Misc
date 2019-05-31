""" A pytorch implementation of the Jaccard loss in 
    UnitBox: An Advanced Object Detection Network. (2016) 
    https://arxiv.org/pdf/1608.01471
"""
import torch 
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F 

# for numerical stability
_IOULOSS_EPSILON = 1e-8


class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels, weight=None, reduction='mean'):

        """
        :param preds:   the estimated positions,    [batch_size, ]
        :param labels:  the ground-truth positions, []  
        """
        assert preds.shape == labels.shape, \
        "input argument preds should have the same dimension as input labels {}, but {} was given instead".format(labels.shape, preds.shape)
        reduction_enum = F._Reduction.get_enum(reduction)

        # prediction values
        xt, xl, xb, xr = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]

        # ground truth values
        gt, gl, gb, gr = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3]

        # prediction area
        X = (xt + xb) * (xl + xr)
        # ground truth area
        G = (gt + gb) * (gl + gr)

        Ih = torch.min(xt, gt) + torch.min(xb, gb)
        Iw = torch.min(xl, gl) + torch.min(xr, gr)

        # intersection, union and IoU. plus one to avoid loss explotion.
        I = Ih * Iw + 1
        U = X + G - I + _IOULOSS_EPSILON + 1

        IoU = I / U

        loss = -torch.log(IoU)

        if weight is not None and weight.sum() > 0:
            assert weight.shape == loss.shape, \
            "The weight should have the same dimension as losses {}, but has {} instead".format(loss.shape, weight.shape)
            loss = (loss * weight)/weight.sum() 

        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()
