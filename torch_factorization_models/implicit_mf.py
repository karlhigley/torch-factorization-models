import logging
from argparse import ArgumentParser
from math import sqrt

import pytorch_lightning as pl
import torch as th
from pytorch_lightning.core.decorators import auto_move_data
from torch_optim_sparse import SparserAdamW

from torch_factorization_models.losses import resolve_loss

logger = logging.getLogger("matrix-factorization")


class ImplicitMatrixFactorization(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.loss_fn = resolve_loss(hparams.loss)
        self.learning_rate = hparams.learning_rate
        self.weight_decay = hparams.weight_decay
        self.use_biases = hparams.use_biases

        self.user_embeddings = th.nn.Embedding(
            hparams.num_users, hparams.embedding_dim, sparse=True
        )

        self.item_embeddings = th.nn.Embedding(
            hparams.num_items, hparams.embedding_dim, sparse=True
        )

        self.global_bias = th.nn.Embedding(1, 1, sparse=True)
        self.global_bias_idx = th.LongTensor([0]).to(device=self.device)

        init_std = 1.0 / sqrt(hparams.embedding_dim)

        th.nn.init.normal_(self.user_embeddings.weight, 0, init_std)
        th.nn.init.normal_(self.item_embeddings.weight, 0, init_std)
        th.nn.init.normal_(self.global_bias.weight, 0, init_std)

        if self.use_biases:
            self.user_biases = th.nn.Embedding(hparams.num_users, 1, sparse=True)
            self.item_biases = th.nn.Embedding(hparams.num_items, 1, sparse=True)

            th.nn.init.normal_(self.user_biases.weight, 0, init_std)
            th.nn.init.normal_(self.item_biases.weight, 0, init_std)

    def configure_optimizers(self):
        return SparserAdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def forward(
        self, user_vectors, item_vectors, user_biases, item_biases, global_bias
    ):
        dots = (user_vectors * item_vectors).sum(dim=1)

        if self.use_biases:
            biases = user_biases + item_biases + global_bias
        else:
            biases = global_bias

        return th.sigmoid(dots + biases)

    def batch_loss(self, batch, batch_idx):
        user_ids = batch["user_ids"]
        item_ids = batch["item_ids"]

        neg_item_ids = th.randint_like(
            item_ids,
            low=0,
            high=self.item_embeddings.num_embeddings,
            dtype=th.int64,
            device=self.device,
        )

        user_vectors = self.user_embeddings(user_ids).squeeze()
        pos_item_vectors = self.item_embeddings(item_ids).squeeze()
        neg_item_vectors = self.item_embeddings(neg_item_ids).squeeze()

        global_bias = self.global_bias(th.tensor(0, device=self.device)).squeeze()

        if self.use_biases:
            user_biases = self.user_biases(user_ids).squeeze()
            pos_item_biases = self.item_biases(item_ids).squeeze()
            neg_item_biases = self.item_biases(neg_item_ids).squeeze()
        else:
            user_biases = pos_item_biases = neg_item_biases = None

        pos_preds = self.forward(
            user_vectors, pos_item_vectors, user_biases, pos_item_biases, global_bias
        )
        neg_preds = self.forward(
            user_vectors, neg_item_vectors, user_biases, neg_item_biases, global_bias
        )

        return self.loss_fn(pos_preds, neg_preds)

    def on_train_start(self):
        # Logger can't be used in __init__, so log the hyperparams here
        self.logger.log_hyperparams(self.hparams)

    def epoch_loss(self, outputs):
        losses = th.cat([o["loss"].reshape(-1) for o in outputs]).flatten()
        return losses.mean()

    def training_step(self, batch, batch_idx):
        loss = self.batch_loss(batch, batch_idx)
        result = pl.TrainResult(minimize=loss)
        result.log("training_loss", loss, on_epoch=True)
        return result

    @th.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.batch_loss(batch, batch_idx)
        result = pl.EvalResult(loss)
        result.log("tuning_loss", loss, on_step=False, on_epoch=True)
        return result

    @th.no_grad()
    def test_step(self, batch, batch_idx):
        loss = self.batch_loss(batch, batch_idx)
        result = pl.EvalResult(loss)
        result.log("testing_loss", loss, on_step=False, on_epoch=True)
        return result

    @auto_move_data
    @th.no_grad()
    def similar_to_users(self, user_ids, k=10):
        user_ids = user_ids - 1

        query_vectors = self.user_embeddings(user_ids).squeeze()
        query_biases = self.user_biases(user_ids).squeeze() if self.use_biases else None

        return self.similar_to_vectors(query_vectors, query_biases, k)

    @auto_move_data
    @th.no_grad()
    def similar_to_items(self, item_ids, k=10):
        item_ids = item_ids - 1

        query_vectors = self.item_embeddings(item_ids).squeeze()
        query_biases = self.item_biases(item_ids).squeeze() if self.use_biases else None

        return self.similar_to_vectors(query_vectors, query_biases, k)

    @auto_move_data
    @th.no_grad()
    def similar_to_vectors(self, query_vectors, query_biases, k=10):
        item_vectors = self.item_embeddings.weight.squeeze()
        dots = query_vectors.mm(item_vectors.t())

        if self.use_biases:
            item_biases = self.item_biases.weight.squeeze()

            biases = query_biases.expand(
                (item_vectors.shape[0], query_biases.shape[0])
            ).t()
            biases += item_biases.expand((query_vectors.shape[0], item_biases.shape[0]))
            biases += self.global_bias(self.global_bias_idx).squeeze()
        else:
            biases = self.global_bias(self.global_bias_idx).squeeze()

        scores = th.sigmoid(dots + biases).detach()

        return th.topk(scores, k)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embedding_dim", default=32, type=int)
        parser.add_argument("--num_users", default=1, type=int)
        parser.add_argument("--num_items", default=1, type=int)

        biases_parser = parser.add_mutually_exclusive_group(required=False)
        biases_parser.add_argument("--biases", dest="use_biases", action="store_true")
        biases_parser.add_argument(
            "--no-biases", dest="use_biases", action="store_false"
        )
        parser.set_defaults(use_biases=False)

        # training specific (for this model)
        parser.add_argument("--learning_rate", default=1e-2, type=float)
        parser.add_argument("--weight_decay", default=1e-2, type=float)
        parser.add_argument("--loss", default="pointwise", type=str)

        return parser