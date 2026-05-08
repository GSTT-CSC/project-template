import logging

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)


class FocalLoss(torch.nn.Module):
    """
    Multiclass focal loss: FL(p_t) = -(1 - p_t)^gamma * CE(p_t)
    Ref: https://arxiv.org/abs/1708.02002.
    Supports hard integer labels and soft labels (e.g. from Mixup).
    Class weights are applied via the CE term.
    :param gamma: focusing parameter; gamma=0 reduces to CE
    :param weight: optional per-class weights tensor (same as CrossEntropyLoss weight)
    :param label_smoothing: label smoothing factor applied to CE
    """
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none',
                                  label_smoothing=self.label_smoothing)
        p_softmax = F.softmax(input, dim=1)
        if target.dim() == 1:
            # Hard labels: p_t = probability of ground truth class
            pt = p_softmax.gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            # Soft labels (e.g. Mixup): expected p_t = sum(target * p_softmax)
            pt = (target * p_softmax).sum(dim=1)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


def get_loss_function(name, weight=None, label_smoothing=0.0):
    """
    Returns an instantiated loss function by name.
    Class weights are passed through to all loss functions that support them.
    :param name: loss function name, one of: CrossEntropyLoss, FocalLoss
    :param weight: optional per-class weights tensor
    :param label_smoothing: label smoothing factor
    :return: instantiated loss function
    """
    if name == "CrossEntropyLoss":
        return CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    elif name == "FocalLoss":
        return FocalLoss(gamma=2.0, weight=weight, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown loss function: '{name}'. Choose from: CrossEntropyLoss, FocalLoss")