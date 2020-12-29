import enum

import torch as th

LOGISTIC_COEFF = 1.0 / th.log(th.tensor(2.0))
INFINITY = th.tensor(float("inf"))


@enum.unique
class LossFunction(enum.Enum):
    LOGISTIC = "logistic"
    CROSSENTROPY = "crossentropy"
    BPR = "bpr"
    HINGE = "hinge"
    WARP = "warp"


def select_loss(name, num_items):
    enum = LossFunction(name)
    if enum == LossFunction.LOGISTIC:
        return logistic_loss
    elif enum == LossFunction.CROSSENTROPY:
        return cross_entropy_loss
    elif enum == LossFunction.BPR:
        return bpr_loss
    elif enum == LossFunction.HINGE:
        return hinge_loss
    elif enum == LossFunction.WARP:
        return build_warp_loss(num_items)
    else:
        raise ValueError(f"Unknown loss function '{name}'")


def logistic_loss(pos_preds, neg_preds, truncate_at=INFINITY):
    raw_losses = th.log1p(th.exp(-pos_preds)) + th.log1p(th.exp(neg_preds))
    return th.clamp(raw_losses * LOGISTIC_COEFF, 0.0, truncate_at)


def cross_entropy_loss(pos_preds, neg_preds):
    def bce(logits, label):
        return (
            th.clamp(logits, min=0) - logits * label + th.log1p(th.exp(-th.abs(logits)))
        )

    losses = bce(pos_preds, 1.0) + bce(neg_preds, 0.0)
    return losses


def bpr_loss(pos_preds, neg_preds):
    losses = -th.log(th.sigmoid(pos_preds - neg_preds))
    return losses


def hinge_loss(pos_preds, neg_preds, margin=0.1):
    distances = th.sigmoid(pos_preds) - th.sigmoid(neg_preds)
    raw_losses = th.max(margin - distances, th.zeros_like(distances))

    return raw_losses / margin


def build_warp_loss(num_items):
    def warp_loss(pos_preds, neg_preds, margin=0.1):
        raw_losses = hinge_loss(pos_preds, neg_preds, margin)

        num_samples = neg_preds.shape[1]
        num_impostors = (raw_losses != 0.0).sum(dim=1)
        weights = th.log(1.0 + (num_impostors * num_items) // num_samples)

        return (raw_losses * weights.unsqueeze(1)) / margin

    return warp_loss
