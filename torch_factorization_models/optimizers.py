import enum

import torch as th
import torch_optim_sparse
from torch_optim_sparse import convert_lr


@enum.unique
class Optimizer(enum.Enum):
    SGD = "sgd"
    SPARSE_ADAM = "sparse_adam"
    SPARSER_SGD = "sparser_sgd"
    SPARSER_SGD_M = "sparser_sgd_m"
    SPARSER_SGDW = "sparser_sgdw"
    SPARSER_SGDW_M = "sparser_sgdw_m"
    SPARSER_ADAM = "sparser_adam"
    SPARSER_ADAMW = "sparser_adamw"


def build_optimizer(params, learning_rate, hparams):
    wd = hparams.weight_decay
    eff_lr = learning_rate
    opt_name = hparams.optimizer

    bs = hparams.batch_size
    m = hparams.momentum
    b1 = hparams.beta1
    b2 = hparams.beta2

    opt = Optimizer(opt_name)

    if opt in [Optimizer.SPARSER_SGDW_M, Optimizer.SPARSER_SGD_M]:
        adj_lr = convert_lr(eff_lr, momentum=m, batch_size=bs)
        b1 = 0.0
        b2 = 0.0
    elif opt in [Optimizer.SPARSER_ADAM, Optimizer.SPARSER_ADAMW]:
        adj_lr = convert_lr(eff_lr, beta1=b1, beta2=b2, batch_size=bs)
        m = 0.0
    else:
        adj_lr = convert_lr(eff_lr, batch_size=bs)
        m = 0.0
        b1 = 0.0
        b2 = 0.0

    hparams.adjusted_lr = adj_lr
    hparams.momentum = m
    hparams.beta1 = b1
    hparams.beta2 = b2

    if opt == Optimizer.SGD:
        return th.optim.SGD(params, lr=adj_lr, weight_decay=wd)
    elif opt == Optimizer.SPARSE_ADAM:
        return th.optim.SparseAdam(params, lr=adj_lr)
    elif opt == Optimizer.SPARSER_SGD:
        return torch_optim_sparse.SparserSGD(params, lr=adj_lr, weight_decay=wd)
    elif opt == Optimizer.SPARSER_SGD_M:
        return torch_optim_sparse.SparserSGD(
            params, lr=adj_lr, weight_decay=wd, momentum=m
        )
    elif opt == Optimizer.SPARSER_SGDW:
        return torch_optim_sparse.SparserSGDW(params, lr=adj_lr, weight_decay=wd)
    elif opt == Optimizer.SPARSER_SGDW_M:
        return torch_optim_sparse.SparserSGDW(
            params, lr=adj_lr, weight_decay=wd, momentum=m
        )
    elif opt == Optimizer.SPARSER_ADAM:
        return torch_optim_sparse.SparserAdam(
            params, lr=adj_lr, betas=(b1, b2), weight_decay=wd
        )
    elif opt == Optimizer.SPARSER_ADAMW:
        return torch_optim_sparse.SparserAdamW(
            params, lr=adj_lr, betas=(b1, b2), weight_decay=wd
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
