from argparse import ArgumentParser

import pytest
import pytorch_lightning as pl
import torch as th
from hypothesis import strategies as strat
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch_factorization_models.implicit_mf import ImplicitMatrixFactorization


@strat.composite
def raw_predictions(draw):
    """Test case generation strategy for raw positive and negative predictions"""
    # batch_size = draw(strat.integers(min_value=1, max_value=1024))
    batch_size = 32

    positives_list = strat.lists(
        strat.floats(min_value=-100, max_value=100),
        min_size=batch_size,
        max_size=batch_size,
    )

    negatives_list = strat.lists(
        strat.floats(min_value=-100, max_value=100),
        min_size=batch_size,
        max_size=batch_size,
    )

    return (draw(positives_list), draw(negatives_list))


class TestDataset(th.utils.data.Dataset):
    def __init__(self, num_examples):
        self.num_examples = num_examples

    @property
    def num_users(self):
        return self.num_examples

    @property
    def num_items(self):
        return self.num_examples

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        return {
            "user_ids": th.tensor([index]),
            "item_ids": th.tensor([index]),
            "targets": th.tensor([1.0]),
        }


class TestDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=1):
        super().__init__()
        self.dataset = TestDataset(100)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.training = self.dataset
        self.tuning = self.dataset
        self.testing = self.dataset

    def train_dataloader(self):
        return DataLoader(
            self.training,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.tuning,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testing,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        # DATASET specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", default=8, type=int)
        parser.add_argument("--num_workers", default=1, type=int)

        return parser


@pytest.fixture
def initialized_model():
    args = [
        "--num_users",
        "100",
        "--num_items",
        "100",
    ]

    parser = ArgumentParser(add_help=False)
    parser = ImplicitMatrixFactorization.add_model_specific_args(parser)
    hparams = parser.parse_args(args)
    model = ImplicitMatrixFactorization(hparams)

    return model


@pytest.fixture(scope="module")
def trained_model():
    args = [
        "--num_users",
        str(100),
        "--num_items",
        str(100),
        "--max_epochs",
        str(100),
    ]

    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser = ImplicitMatrixFactorization.add_model_specific_args(parser)
    parser = TestDataModule.add_dataset_specific_args(parser)
    parsed_args = parser.parse_args(args)
    model = ImplicitMatrixFactorization(parsed_args)

    test_dataset = TestDataModule(parsed_args.batch_size, parsed_args.num_workers)

    trainer = Trainer.from_argparse_args(parsed_args)
    trainer.fit(model, test_dataset)

    return model
