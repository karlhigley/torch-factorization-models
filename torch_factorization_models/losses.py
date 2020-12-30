import enum

import torch as th


@enum.unique
class LossFunction(enum.Enum):
    BCE = "bce"
    BPR = "bpr"
    HINGE = "hinge"
    WARP = "warp"


def select_loss(name, num_items, margin=0.1):
    enum = LossFunction(name)
    if enum == LossFunction.BCE:
        return binary_cross_entropy_loss
    elif enum == LossFunction.BPR:
        return bpr_loss
    elif enum == LossFunction.HINGE:
        return build_hinge_loss(margin)
    elif enum == LossFunction.WARP:
        return build_warp_loss(num_items, margin)
    else:
        raise ValueError(f"Unknown loss function '{name}'")


def _stable_bce(logits, label):
    return th.clamp(logits, min=0) - logits * label + th.log1p(th.exp(-th.abs(logits)))


def binary_cross_entropy_loss(pos_preds, neg_preds):
    losses = _stable_bce(pos_preds, 1.0) + _stable_bce(neg_preds, 0.0)
    return losses


def bpr_loss(pos_preds, neg_preds):
    losses = _stable_bce(pos_preds - neg_preds, 1.0)
    return losses


def build_hinge_loss(margin):
    def hinge_loss(pos_preds, neg_preds):
        distances = th.sigmoid(pos_preds) - th.sigmoid(neg_preds)
        raw_losses = th.clamp(margin - distances, min=0)

        return raw_losses / margin

    return hinge_loss


def build_warp_loss(num_items, margin):
    hinge_loss = build_hinge_loss(margin)

    def warp_loss(pos_preds, neg_preds):
        raw_losses = hinge_loss(pos_preds, neg_preds)

        num_samples = neg_preds.shape[1]
        num_impostors = (raw_losses != 0.0).sum(dim=1)
        weights = th.log1p((num_impostors * num_items) // num_samples)

        return (raw_losses * weights.unsqueeze(1)) / margin

    return warp_loss
