from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch as th
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from torch_factorization_models.implicit_mf import ImplicitMatrixFactorization
from torch_factorization_models.movielens import MovielensDataModule

# sets seeds for numpy, torch, etc...
# must do for DDP to work well
seed_everything(42)


def main(args):
    # Load the dataset
    movielens = MovielensDataModule(
        args.data_dir,
        args.filename,
        args.split,
        args.threshold,
        args.negatives,
        args.batch_size,
        args.num_workers,
    )

    args.num_items = movielens.dataset.num_items
    args.num_users = movielens.dataset.num_users

    # Set up the model and logger
    model = ImplicitMatrixFactorization(hparams=args)

    if th.cuda.is_available() and args.gpus > 0:
        model.cuda()

    wandb_logger = WandbLogger(project="torch-factorization-models")
    wandb_logger.watch(model, log="all", log_freq=100)

    if args.early_stopping:
        args.early_stopping = EarlyStopping(monitor="tuning_loss")

    # Most basic trainer, uses good defaults
    trainer = Trainer.from_argparse_args(
        args,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        early_stop_callback=args.early_stopping,
    )

    if args.use_lr_finder:
        movielens.setup()

        lr_finder = trainer.lr_find(
            model,
            train_dataloader=movielens.train_dataloader(),
            val_dataloaders=[movielens.val_dataloader()],
            early_stop_threshold=None,
            min_lr=1e-6,
            max_lr=5e-1,
        )

        lr_finder.plot(suggest=True)
        plt.show(block=True)

    else:
        trainer.fit(model, movielens)

        # Save the model
        th.save(model.state_dict(), Path(wandb_logger.experiment.dir) / "model.pt")


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    # Add args from trainer class
    parser = Trainer.add_argparse_args(parser)

    # Add custom trainer arguments
    lr_finding_parser = parser.add_mutually_exclusive_group(required=False)
    lr_finding_parser.add_argument(
        "--find_lr", dest="use_lr_finder", action="store_true"
    )
    parser.set_defaults(use_lr_finder=False)

    early_stopping_parser = parser.add_mutually_exclusive_group(required=False)
    early_stopping_parser.add_argument(
        "--stop_early", dest="early_stopping", action="store_true"
    )
    parser.set_defaults(early_stopping=False)

    # Give the model and dataset a chance to add their own params
    parser = ImplicitMatrixFactorization.add_model_specific_args(parser)
    parser = MovielensDataModule.add_dataset_specific_args(parser)

    # parse params
    args = parser.parse_args()

    main(args)
