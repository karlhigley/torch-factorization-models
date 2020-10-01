import enum

import torch as th


@enum.unique
class LossFunction(enum.Enum):
    BPR = "bpr"
    POINTWISE = "pointwise"


def resolve_loss(name):
    enum = LossFunction(name)
    if enum == LossFunction.BPR:
        return bpr_loss
    elif enum == LossFunction.POINTWISE:
        return pointwise_loss
    else:
        raise ValueError(f"Unknown loss function '{name}'")


def bpr_loss(pos_preds, neg_preds):
    losses = 1.0 - th.sigmoid(pos_preds - neg_preds)
    return losses.mean()


def pointwise_loss(pos_preds, neg_preds):
    pos_loss = 1.0 - pos_preds
    neg_loss = neg_preds

    return (pos_loss + neg_loss).mean()
