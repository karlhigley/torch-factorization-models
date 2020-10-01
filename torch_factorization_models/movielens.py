import logging
from argparse import ArgumentParser
from math import floor
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch as th
from torch.utils.data import DataLoader, random_split

logger = logging.getLogger("movielens-dataset")


class MovielensDataset(th.utils.data.Dataset):
    def __init__(self, path, filename="ratings.csv", threshold=4.0):
        interactions_path = Path(path) / filename
        interactions = pd.read_csv(interactions_path)

        interactions = interactions.rename(
            columns={"userId": "user_id", "movieId": "item_id", "rating": "target"}
        )

        interactions["user_id"] -= 1
        interactions["item_id"] -= 1

        kept_rows = interactions[interactions["target"] >= threshold]
        interactions = kept_rows.copy()
        interactions.assign(target=1.0)

        self.user_ids = th.tensor(interactions["user_id"].values, dtype=th.int64)
        self.item_ids = th.tensor(interactions["item_id"].values, dtype=th.int64)
        self.targets = th.tensor(interactions["target"].values, dtype=th.float64)

    @property
    def num_users(self):
        return int(self.user_ids.max()) + 1

    @property
    def num_items(self):
        return int(self.item_ids.max()) + 1

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return {
            "user_ids": self.user_ids[index],
            "item_ids": self.item_ids[index],
            "targets": self.targets[index],
        }


class MovielensDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=".", batch_size=64, num_workers=1):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = MovielensDataset(self.data_dir)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        num_examples = len(self.dataset)
        tune_examples = test_examples = floor(0.1 * num_examples)
        train_examples = num_examples - test_examples - tune_examples

        splits = (
            train_examples,
            tune_examples,
            test_examples,
        )

        self.training, self.tuning, self.testing = random_split(self.dataset, splits)

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
        parser.add_argument("--data_dir", default=".", type=str)
        parser.add_argument("--batch_size", default=512, type=int)
        parser.add_argument("--num_workers", default=1, type=int)

        return parser
