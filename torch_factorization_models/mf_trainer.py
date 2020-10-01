import logging
from argparse import ArgumentParser
from pathlib import Path

import torch as th
from pytorch_lightning import Trainer, seed_everything


from torch_factorization_models.matrix_factorization import ImplicitMatrixFactorization
from torch_factorization_models.movielens import MovielensDataModule

# sets seeds for numpy, torch, etc...
# must do for DDP to work well
seed_everything(42)

logger = logging.getLogger("implicit-mf-trainer")


def main(args):
    # Load the dataset
    movielens = MovielensDataModule(args.data_dir, args.batch_size, args.num_workers)

    args.num_items = movielens.dataset.num_items
    args.num_users = movielens.dataset.num_users

    logger.warning(f"Num items: {args.num_items} Num users: {args.num_users}")

    # Set up the model and logger
    model = ImplicitMatrixFactorization(hparams=args)

    # Most basic trainer, uses good defaults
    trainer = Trainer.from_argparse_args(args, check_val_every_n_epoch=5)

    trainer.fit(model, movielens)

    # Save the model
    th.save(model.state_dict(), Path(".") / "model.pt")


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    # Add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # Give the model and dataset a chance to add their own params
    parser = ImplicitMatrixFactorization.add_model_specific_args(parser)
    parser = MovielensDataModule.add_dataset_specific_args(parser)

    # parse params
    args = parser.parse_args()

    main(args)
