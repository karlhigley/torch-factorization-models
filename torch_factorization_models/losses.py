import enum

import torch as th


@enum.unique
class LossFunction(enum.Enum):
    BCE = "bce"
    BPR = "bpr"
    HINGE = "hinge"
    WARP = "warp"


def select_loss(name, num_items, margin=0.1):
    """Select/construct a loss function by name

    Args:
        name (string): Short name of the desired loss
        num_items (int): Total number of items in the dataset (used for WARP loss)
        margin (float, optional): Margin for hinge and WARP loss. Defaults to 0.1.

    Raises:
        ValueError: When name doesn't correspond to a known loss function

    Returns:
        function: Loss function for corresponding positive and negative examples
    """
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
    """Numerically stable implementation of binary cross-entropy

    This avoids problems at the boundaries when the output of a sigmoid function is
    passed into a log.

    Args:
        logits (th.Tensor): Model outputs before activation function
        label (float): Label for the corresponding examples

    Returns:
        th.Tensor: Binary cross-entropies computed from logits and label
    """
    return th.clamp(logits, min=0) - logits * label + th.log1p(th.exp(-th.abs(logits)))


def binary_cross_entropy_loss(pos_preds, neg_preds):
    """Binary cross-entropy loss for paired positive and negative examples

    Args:
        pos_preds (th.Tensor): Logits for positive examples
        neg_preds (th.Tensor): Logits for negative examples

    Returns:
        th.Tensor: Computed losses for each example pair
    """
    losses = _stable_bce(pos_preds, 1.0) + _stable_bce(neg_preds, 0.0)
    return losses


def bpr_loss(pos_preds, neg_preds):
    """Bayesian Personalized Ranking loss for paired positive/negative examples

    This would naively be computed as -th.log(th.sigmoid(pos_preds - neg_preds)),
    but that has similar numerical instability problems as the naive computation
    of binary cross-entropy. Since it's equivalent to the positive example term
    of BCE loss, we can re-use the stable implementation of BCE here and treat
    each pair of positive and negative predictions as a single positive example.

    Args:
        pos_preds (th.Tensor): Logits for positive examples
        neg_preds (th.Tensor): Logits for negative examples

    Returns:
        th.Tensor: Computed losses for each example pair
    """
    losses = _stable_bce(pos_preds - neg_preds, 1.0)
    return losses


def build_hinge_loss(margin):
    """Builds a hinge loss function w/ the supplied margin

    Args:
        margin (float): Hinge loss margin
    """

    def hinge_loss(pos_preds, neg_preds):
        """Ranking hinge loss

        Args:
            pos_preds (th.Tensor): Logits for positive examples
            neg_preds (th.Tensor): Logits for negative examples

        Returns:
            th.Tensor: Computed losses for each example pair
        """
        distances = th.sigmoid(pos_preds) - th.sigmoid(neg_preds)
        raw_losses = th.clamp(margin - distances, min=0.0)

        return raw_losses

    return hinge_loss


def build_warp_loss(num_items, margin):
    """Builds a WARP loss function w/ supplied margin and number of items

    Args:
        num_items (int): Total number of items in the dataset
        margin (float): Hinge loss margin (between 0.0 and 1.0)

    Returns:
        th.Tensor: Computed losses for each example pair
    """
    hinge_loss = build_hinge_loss(margin)
    normalization_factor = margin * th.log1p(th.tensor(num_items, dtype=th.float32))

    def warp_loss(pos_preds, neg_preds):
        """Weighted Approximate Rank Pairwise loss

        Args:
            pos_preds (th.Tensor): Logits for positive examples
            neg_preds (th.Tensor): Logits for negative examples

        Returns:
            th.Tensor: Computed losses for each example pair
        """
        raw_losses = hinge_loss(pos_preds, neg_preds)

        num_samples = neg_preds.shape[1]
        num_impostors = (raw_losses != 0.0).sum(dim=1).to(dtype=th.float32)
        weights = (
            th.log1p((num_impostors * num_items) // num_samples) / normalization_factor
        )

        return raw_losses * weights.unsqueeze(1)

    return warp_loss
