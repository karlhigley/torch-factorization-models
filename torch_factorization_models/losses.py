import enum

import torch as th

LOGISTIC_COEFF = 1.0 / th.log(th.tensor(2.0))
INFINITY = th.tensor(float("inf"))


@enum.unique
class LossFunction(enum.Enum):
    LOGISTIC = "logistic"
    BPR = "bpr"
    HINGE = "hinge"
    WARP = "warp"


def select_loss(name):
    enum = LossFunction(name)
    if enum == LossFunction.LOGISTIC:
        return logistic_loss
    elif enum == LossFunction.BPR:
        return bpr_loss
    elif enum == LossFunction.HINGE:
        return hinge_loss
    elif enum == LossFunction.WARP:
        return warp_loss
    else:
        raise ValueError(f"Unknown loss function '{name}'")


def logistic_loss(pos_preds, neg_preds, truncate_at=INFINITY):
    raw_losses = th.log1p(th.exp(-pos_preds)) + th.log1p(th.exp(neg_preds))
    return th.clamp(raw_losses * LOGISTIC_COEFF, 0.0, truncate_at)


def bpr_loss(pos_preds, neg_preds):
    losses = -th.log(th.sigmoid(pos_preds - neg_preds))
    return losses


def hinge_loss(pos_preds, neg_preds, margin=0.1):
    distances = th.sigmoid(pos_preds) - th.sigmoid(neg_preds)
    raw_losses = th.max(margin - distances, th.zeros_like(distances))
    margin_coeff = th.empty_like(raw_losses).fill_(1.0 / margin)

    return raw_losses * margin_coeff


def warp_loss(pos_preds, neg_preds, num_items, margin=0.1):
    num_samples = neg_preds.shape[1]
    raw_losses = hinge_loss(pos_preds.expand(neg_preds.shape), neg_preds, margin)
    num_impostors = (raw_losses != 0.0).sum(dim=1)
    weights = th.log(1 + th.floor((num_impostors * num_items).true_divide(num_samples)))
    margin_coeff = th.empty_like(raw_losses).fill_(1.0 / margin)

    return raw_losses * weights.reshape(-1, 1).expand(raw_losses.shape) * margin_coeff
