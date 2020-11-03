import logging
from argparse import ArgumentParser
from collections import defaultdict
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch as th
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Binarizer, OrdinalEncoder
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger("movielens-dataset")


class MovielensDataset(th.utils.data.Dataset):
    def __init__(self, path, filename="ratings.csv", threshold=4.0):
        interactions_path = Path(path) / filename
        interactions = pd.read_csv(interactions_path)

        interactions = interactions.rename(
            columns={"userId": "user_id", "movieId": "item_id", "rating": "target"}
        )

        interactions = interactions[interactions["target"] >= threshold].copy()

        xformer = ColumnTransformer(
            [
                ("user_id", OrdinalEncoder(dtype=np.int64), ["user_id"]),
                ("item_id", OrdinalEncoder(dtype=np.int64), ["item_id"]),
                ("target", Binarizer(threshold=threshold), ["target"]),
            ],
            remainder="passthrough",
        )

        interactions = pd.DataFrame(
            xformer.fit_transform(interactions), columns=interactions.columns
        )

        self.preprocessor = xformer
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


class MovielensEvalDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.interactions = self._build_interaction_vectors(subset)

    def __len__(self):
        return len(self.interactions.keys())

    def __getitem__(self, index):
        user_id = int(index)
        user_interactions = self.interactions[user_id]

        return {
            "user_ids": th.tensor([user_id]),
            "interactions": user_interactions,
        }

    def _build_interaction_vectors(self, subset):
        num_users = subset.dataset.num_users
        num_items = subset.dataset.num_items

        interactions = defaultdict(
            lambda: self._empty_sparse_vector(num_users, num_items)
        )

        # Create a sparse vector for each user's item interactions
        current_user_id = 0
        current_item_ids = []

        for index in sorted(subset.indices):
            example = subset.dataset[index]
            user_id = example["user_ids"]
            item_id = example["item_ids"]

            if user_id != current_user_id:
                # Build sparse vector with accumulated interactions
                interactions[int(user_id)] = self._sparse_vector(
                    current_user_id, current_item_ids, num_users, num_items,
                )

                # Clear accumulators for the next user
                current_user_id = user_id
                current_item_ids = []

            # Add the current item to the interactions for this user
            current_item_ids.append(item_id)

        return interactions

    def _sparse_vector(self, user_id, item_ids, num_users, num_items):
        item_indices = th.tensor(item_ids, dtype=th.int64)
        user_indices = th.empty_like(item_indices, dtype=th.int64).fill_(user_id)
        item_labels = th.ones_like(item_indices, dtype=th.float64)

        return th.sparse.FloatTensor(
            th.stack([user_indices, item_indices]), item_labels, (num_users, num_items)
        )

    def _empty_sparse_vector(self, num_users, num_items):
        return th.sparse.FloatTensor(
            th.tensor([[], []], dtype=th.int64),
            th.tensor([], dtype=th.float64),
            (num_users, num_items),
        )


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

    def val_dataloader(self, by_user=False):
        if by_user:
            return DataLoader(
                MovielensEvalDataset(self.tuning), batch_size=self.batch_size,
            )
        else:
            return DataLoader(
                self.tuning,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=True,
            )

    def test_dataloader(self, by_user=False):
        if by_user:
            return DataLoader(
                MovielensEvalDataset(self.testing), batch_size=self.batch_size,
            )
        else:
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
