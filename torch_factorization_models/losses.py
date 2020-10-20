import enum

import torch as th


@enum.unique
class LossFunction(enum.Enum):
    BPR = "bpr"
    HINGE = "hinge"
    WARP = "warp"


def select_loss(name):
    enum = LossFunction(name)
    if enum == LossFunction.BPR:
        return bpr_loss
    elif enum == LossFunction.HINGE:
        return hinge_loss
    else:
        raise ValueError(f"Unknown loss function '{name}'")


def bpr_loss(pos_preds, neg_preds):
    distances = th.sigmoid(pos_preds - neg_preds)
    losses = -th.log(distances)
    return losses


def hinge_loss(pos_preds, neg_preds, margin=1.0):
    distances = th.sigmoid(pos_preds) - th.sigmoid(neg_preds)
    losses = th.max(margin - distances, th.zeros_like(distances))

    return losses


def warp_loss(pos_preds, neg_preds, num_items, margin=1.0):
    num_samples = neg_preds.shape[1]
    raw_losses = hinge_loss(pos_preds.expand(neg_preds.shape), neg_preds, margin)
    num_impostors = (raw_losses != 0.0).sum(dim=1)
    weights = th.log(1 + th.floor((num_impostors * num_items).true_divide(num_samples)))

    return raw_losses * weights.reshape(-1, 1).expand(raw_losses.shape)
