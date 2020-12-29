import enum
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch as th
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Binarizer, OrdinalEncoder
from torch._utils import _accumulate
from torch.utils.data import DataLoader, Dataset, Subset, random_split


@enum.unique
class DataSplitting(enum.Enum):
    RANDOM = "random"
    TEMPORAL = "temporal"


def sequential_split(dataset, lengths):
    r"""
    Sequentially split a dataset into non-overlapping new datasets of given lengths.

    >>> sequential_split(range(10), [3, 7])

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = list(range(sum(lengths)))
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


class MovielensDataset(th.utils.data.Dataset):
    def __init__(self, path, filename, threshold, negatives):
        interactions_path = Path(path) / filename
        interactions = pd.read_csv(interactions_path)

        interactions = interactions.rename(
            columns={"userId": "user_id", "movieId": "item_id", "rating": "target"}
        )

        interactions = (
            interactions[interactions["target"] > threshold]
            .copy()
            .sort_values(by=["timestamp"])
        )

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

        self.num_users = int(interactions["user_id"].max()) + 1
        self.num_items = int(interactions["item_id"].max()) + 1

        self.data = th.tensor(
            interactions[["user_id", "item_id", "target"]].to_numpy(), dtype=th.int64
        )

        self.negatives = negatives

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        return (self[index] for index in range(len(self)))

    def __getitem__(self, index):
        row = self.data[int(index)]
        user_id = row[0]
        item_id = row[1]
        target = row[2]

        neg_item_ids = th.randint(
            low=0,
            high=self.num_items,
            size=(self.negatives,),
            dtype=th.int64,
            device=self.data.device,
        )

        return {
            "user_ids": user_id,
            "item_ids": item_id,
            "neg_item_ids": neg_item_ids,
            "targets": target,
        }

    def to_(self, *args, **kwargs):
        self.data = self.data.to(*args, **kwargs)


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

    def __iter__(self):
        return (self[index] for index in range(len(self)))

    def __getitem__(self, index):
        user_id = int(index)
        user_interactions = self.interactions[user_id]

        return {
            "user_ids": th.tensor([user_id], device=user_interactions.device),
            "interactions": user_interactions,
        }

    def _build_interaction_vectors(self, dataset, indices, num_users, num_items):
        sorted_indices = sorted(indices)

        user_ids, item_ids, targets = dataset.data[sorted_indices].t()
        device = dataset.data.device

        interactions = defaultdict(
            lambda: self._empty_sparse_vector(num_users, num_items, device)
        )

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
                device,
            )

        return interactions

    def _sparse_vector(self, user_id, item_ids, targets, num_users, num_items, device):
        item_indices = item_ids.to(dtype=th.int64)
        user_indices = th.empty_like(item_indices, dtype=th.int64).fill_(user_id)
        item_labels = targets.to(dtype=th.float64)

        return th.sparse.FloatTensor(
            th.stack([user_indices, item_indices]), item_labels, (num_users, num_items)
        ).to(device=device)

    def _empty_sparse_vector(self, num_users, num_items, device):
        return th.sparse.FloatTensor(
            th.tensor([[], []], dtype=th.int64),
            th.tensor([], dtype=th.float64),
            (num_users, num_items),
        ).to(device=device)


class MovielensDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir=".",
        filename="ratings.csv",
        split="random",
        threshold=3.5,
        negatives=1,
        batch_size=64,
        num_workers=1,
    ):
        super().__init__()
        self.split = split
        self.dataset = MovielensDataset(data_dir, filename, threshold, negatives)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        num_examples = len(self.dataset)
        tune_examples = test_examples = num_examples // 10
        train_examples = num_examples - test_examples - tune_examples

        strategy = DataSplitting(self.split)

        if strategy == DataSplitting.TEMPORAL:
            evaluation_examples = tune_examples + test_examples

            self.training, evaluation = sequential_split(
                self.dataset, [train_examples, evaluation_examples]
            )

            self.tuning, self.testing = random_split(
                evaluation, [tune_examples, test_examples]
            )
        else:  # strategy == DataSplitting.RANDOM
            splits = (
                train_examples,
                tune_examples,
                test_examples,
            )

            self.training, self.tuning, self.testing = random_split(
                self.dataset, splits
            )

    def train_dataloader(self, by_user=False):
        if by_user:
            dataset = MovielensEvalDataset(self.training)
            return DataLoader(dataset, batch_size=self.batch_size)
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
            dataset = MovielensEvalDataset(self.tuning)
            return DataLoader(dataset, batch_size=self.batch_size)
        else:
            return DataLoader(
                self.tuning,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=True,
            )

    def test_dataloader(self, by_user=False):
        if by_user:
            dataset = MovielensEvalDataset(self.testing)
            return DataLoader(dataset, batch_size=self.batch_size)
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
        parser.add_argument("--filename", default="ratings.csv", type=str)
        parser.add_argument("--split", default="random", type=str)
        parser.add_argument("--threshold", default=3.5, type=float)
        parser.add_argument("--negatives", default=1, type=int)
        parser.add_argument("--batch_size", default=512, type=int)
        parser.add_argument("--num_workers", default=1, type=int)

        return parser
