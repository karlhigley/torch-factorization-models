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
        user_id = self.user_ids[index]
        item_id = self.item_ids[index]
        target = self.targets[index]

        neg_item_id = th.randint_like(
            item_id, low=0, high=self.num_items, dtype=th.int64
        )

        return {
            "user_ids": user_id,
            "item_ids": item_id,
            "neg_item_ids": neg_item_id,
            "targets": target,
        }

    def to_(self, *args, **kwargs):
        self.user_ids = self.user_ids.to(*args, **kwargs)
        self.item_ids = self.item_ids.to(*args, **kwargs)
        self.targets = self.targets.to(*args, **kwargs)


class MovielensEvalDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

        self.num_users = subset.dataset.num_users
        self.num_items = subset.dataset.num_items

        self.interactions = self._build_interaction_vectors(
            subset.dataset, subset.indices, self.num_users, self.num_items
        )

    def __len__(self):
        return self.num_users

    def __getitem__(self, index):
        user_id = int(index)
        user_interactions = self.interactions[user_id]

        return {
            "user_ids": th.tensor([user_id], device=user_interactions.device),
            "interactions": user_interactions,
        }

    def _build_interaction_vectors(self, dataset, indices, num_users, num_items):
        sorted_indices = sorted(indices)

        user_ids = dataset.user_ids[sorted_indices]
        item_ids = dataset.item_ids[sorted_indices]
        targets = dataset.targets[sorted_indices]

        default_value = self._empty_sparse_vector(num_users, num_items, user_ids.device)
        interactions = defaultdict(lambda: default_value.clone().detach())

        # Find unique user id values
        unique_user_ids = th.unique(user_ids)

        for n, current_user_id in enumerate(unique_user_ids):
            current_user_indices = user_ids == current_user_id
            current_item_ids = item_ids[current_user_indices]
            current_targets = targets[current_user_indices]
            interactions[int(current_user_id)] = self._sparse_vector(
                current_user_id,
                current_item_ids,
                current_targets,
                num_users,
                num_items,
            )

        return interactions

    def _sparse_vector(self, user_id, item_ids, targets, num_users, num_items):
        item_indices = item_ids.to(dtype=th.int64)
        user_indices = th.empty_like(item_indices, dtype=th.int64).fill_(user_id)
        item_labels = targets.to(dtype=th.float64)

        return th.sparse.FloatTensor(
            th.stack([user_indices, item_indices]), item_labels, (num_users, num_items)
        )

    def _empty_sparse_vector(self, num_users, num_items, device):
        return th.sparse.FloatTensor(
            th.tensor([[], []], dtype=th.int64),
            th.tensor([], dtype=th.float64),
            (num_users, num_items),
        ).to(device=device)


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

    def train_dataloader(self, by_user=False):
        if by_user:
            return DataLoader(
                MovielensEvalDataset(self.training), batch_size=self.batch_size,
            )
        else:
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
