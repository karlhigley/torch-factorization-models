import torch as th
from hypothesis import given
from torch_factorization_models.losses import (
    bpr_loss,
    build_hinge_loss,
    build_warp_loss,
)

from tests.conftest import raw_predictions


@given(raw_predictions())
def test_hinge_loss_range(raw_predictions):
    positives, negatives = raw_predictions

    hinge_loss = build_hinge_loss(1.0)

    loss = hinge_loss(th.tensor(positives), th.tensor(negatives)).mean()
    assert (loss >= 0.0).all()
    assert (loss <= 2.0).all()


@given(raw_predictions())
def test_bpr_loss_range(raw_predictions):
    positives, negatives = raw_predictions

    loss = bpr_loss(th.tensor(positives), th.tensor(negatives)).mean()
    assert (loss >= 0.0).all()


@given(raw_predictions())
def test_warp_loss_range(raw_predictions):
    positives, negatives = raw_predictions

    warp_loss = build_warp_loss(1000, 0.1)

    loss = warp_loss(th.tensor(positives), th.tensor([negatives, negatives])).mean()
    assert (loss >= 0.0).all()
