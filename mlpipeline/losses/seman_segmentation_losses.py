import torch
import torch.nn as nn
import torch.nn.functional as functional
import monai.losses
from einops import rearrange
from torch.nn.modules.loss import _Loss

from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss, TverskyLoss, FocalLoss, LovaszLoss
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss
from monai.losses import DiceCELoss, DiceFocalLoss


class BinaryFocalLoss(nn.Module):
    def __init__(self):
        super(BinaryFocalLoss, self).__init__()

    @staticmethod
    def binary_focal(pred, gt, gamma=2, *args):
        return -gt * torch.log(pred) * torch.pow(1 - pred, gamma)

    def forward(self, pred, gt, gamma=2, eps=1e-6, *args):
        pred = torch.clamp(pred, eps, 1 - eps)
        loss1 = self.binary_focal(pred, gt, gamma=gamma)
        loss2 = self.binary_focal(1 - pred, 1 - gt, gamma=gamma)
        loss = loss1 + loss2
        return loss.mean()


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        loss = self.criterion(pred, gt.squeeze(dim=1).long())
        return loss


class DiceBCE(nn.Module):
    def __init__(self, mode, from_logits, smooth, pos_weight):
        super(DiceBCE, self).__init__()
        self.dice = DiceLoss(mode=mode, from_logits=from_logits, smooth=smooth)
        self.bce = SoftBCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]), smooth_factor=smooth)

    def forward(self, pred, gt):
        loss = self.dice(pred, gt)
        loss = loss + self.bce(pred, gt.float())
        return loss


def create_segmentation_loss(name, **kwargs):
    mode = kwargs["mode"]
    from_logits = kwargs["from_logits"]
    smooth = kwargs["smooth"]
    loss_type = kwargs.get("loss_type", 1)
    pos_weight = kwargs.get("pos_weight", 1.0)

    if name == "dice":
        return DiceLoss(mode=mode, from_logits=from_logits, smooth=smooth)
    elif name == "jaccard":
        return JaccardLoss(mode=mode, from_logits=from_logits, smooth=smooth)
    elif name == "tversky":
        return TverskyLoss(mode=mode, from_logits=from_logits, smooth=smooth)
    elif name == "focal":
        return FocalLoss(mode=mode, **kwargs)
    elif name == "binary-focal":
        return BinaryFocalLoss()
    elif name == "lovasz":
        return LovaszLoss(**kwargs)
    elif name == "bce":
        return SoftBCEWithLogitsLoss()
    elif name == "ce":
        return CrossEntropyLoss()
    elif name == "dicebce":
        return DiceBCE(mode=mode, from_logits=from_logits, smooth=smooth, pos_weight=pos_weight)
    elif name == "dicefocal":
        return DiceFocalLoss(include_background=False)
    else:
        raise ValueError(f'Not support loss {name}.')
