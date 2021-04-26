from argparse import ArgumentParser
from math import sqrt

import pytorch_lightning as pl
import torch as th
from pytorch_lightning.core.decorators import auto_move_data

from torch_factorization_models.losses import select_loss
from torch_factorization_models.optimizers import build_optimizer


class ImplicitMatrixFactorization(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.loss_fn = select_loss(hparams.loss, hparams.num_items, hparams.margin)
        self.learning_rate = hparams.learning_rate
        self.weight_decay = hparams.weight_decay

        self.use_biases = hparams.use_biases
        self.sparse = hparams.optimizer != "sgd"

        self.user_embeddings = th.nn.Embedding(
            hparams.num_users, hparams.embedding_dim, sparse=self.sparse
        )

        self.item_embeddings = th.nn.Embedding(
            hparams.num_items, hparams.embedding_dim, sparse=self.sparse
        )

        # Registering this as a buffer makes model.cuda() work properly
        self.register_buffer("global_bias_idx", th.LongTensor([0], device=self.device))
        self.global_bias = th.nn.Embedding(1, 1, sparse=self.sparse)

        init_std = 1.0 / sqrt(hparams.embedding_dim)

        th.nn.init.normal_(self.user_embeddings.weight, 0, init_std)
        th.nn.init.normal_(self.item_embeddings.weight, 0, init_std)
        th.nn.init.normal_(self.global_bias.weight, 0, init_std)

        self.user_biases = th.nn.Embedding(hparams.num_users, 1, sparse=self.sparse)
        self.item_biases = th.nn.Embedding(hparams.num_items, 1, sparse=self.sparse)

        if self.use_biases:
            th.nn.init.normal_(self.user_biases.weight, 0, init_std)
            th.nn.init.normal_(self.item_biases.weight, 0, init_std)
        else:
            th.nn.init.zeros_(self.user_biases.weight)
            th.nn.init.zeros_(self.item_biases.weight)

    def configure_optimizers(self):
        return build_optimizer(
            list(self.parameters()), self.learning_rate, self.hparams
        )

    def forward(
        self, user_vectors, item_vectors, user_biases, item_biases, global_bias
    ):

        if len(item_vectors.shape) < 3:
            item_vectors = item_vectors.unsqueeze(1)

        if len(user_vectors.shape) < 3:
            user_vectors = user_vectors.unsqueeze(1).expand(item_vectors.shape)

        dots = (user_vectors * item_vectors).sum(dim=2)

        if self.use_biases:
            if len(item_biases.shape) < 3:
                item_biases = item_biases.unsqueeze(1)

            if len(user_biases.shape) < 3:
                user_biases = user_biases.unsqueeze(1).expand(item_biases.shape)

            biases = user_biases + item_biases + global_bias.expand(user_biases.shape)
        else:
            biases = global_bias

        return dots + biases

    def batch_loss(self, batch, batch_idx):
        user_ids = batch["user_ids"]
        item_ids = batch["item_ids"]
        neg_item_ids = batch["neg_item_ids"]

        user_vectors = self.user_embeddings(user_ids)
        pos_item_vectors = self.item_embeddings(item_ids)
        neg_item_vectors = self.item_embeddings(neg_item_ids)

        global_bias = self.global_bias(th.tensor(0, device=self.device)).squeeze()

        if self.use_biases:
            user_biases = self.user_biases(user_ids)
            pos_item_biases = self.item_biases(item_ids)
            neg_item_biases = self.item_biases(neg_item_ids)
        else:
            user_biases = pos_item_biases = neg_item_biases = None

        pos_preds = self.forward(
            user_vectors, pos_item_vectors, user_biases, pos_item_biases, global_bias
        )
        neg_preds = self.forward(
            user_vectors, neg_item_vectors, user_biases, neg_item_biases, global_bias
        )

        return self.loss_fn(pos_preds, neg_preds).mean()

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
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("tuning_loss", loss, on_step=False, on_epoch=True)
        return result

    @th.no_grad()
    def test_step(self, batch, batch_idx):
        loss = self.batch_loss(batch, batch_idx)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("testing_loss", loss, on_step=False, on_epoch=True)
        return result

    @auto_move_data
    @th.no_grad()
    def similarity_to_users(self, user_ids):
        query_vectors = self.user_embeddings(user_ids).squeeze()
        query_biases = (
            self.user_biases(user_ids).squeeze()
            if self.use_biases
            else th.zeros((1, 1), device=self.device)
        )

        query_biases = th.ones_like(query_biases)

        return self.similarity_to_vectors(query_vectors, query_biases)

    @auto_move_data
    @th.no_grad()
    def similarity_to_items(self, item_ids):
        query_vectors = self.item_embeddings(item_ids).squeeze()
        query_biases = (
            self.item_biases(item_ids).squeeze()
            if self.use_biases
            else th.zeros((1, 1), device=self.device)
        )

        return self.similarity_to_vectors(query_vectors, query_biases)

    @auto_move_data
    @th.no_grad()
    def similarity_to_vectors(self, query_vectors, query_biases):
        item_vectors = self.item_embeddings.weight.squeeze()
        item_biases = self.item_biases.weight.squeeze()

        scores = self._similarity_scores(
            query_vectors, query_biases, item_vectors, item_biases
        )

        return scores

    @auto_move_data
    @th.no_grad()
    def _similarity_scores(
        self, query_vectors, query_biases, item_vectors, item_biases
    ):
        dots = query_vectors.mm(item_vectors.t())

        biases = th.zeros(
            (query_vectors.shape[0], item_biases.shape[0]), device=self.device
        )
        biases += self.global_bias(self.global_bias_idx).squeeze()

        if self.use_biases:
            biases += query_biases.expand(
                (item_vectors.shape[0], query_biases.shape[0])
            ).t() * item_biases.expand((query_vectors.shape[0], item_biases.shape[0]))

        scores = th.sigmoid(dots + biases).detach()

        return scores

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
        parser.add_argument("--loss", default="bce", type=str)
        parser.add_argument("--margin", default=0.1, type=float)

        parser.add_argument("--learning_rate", default=1e-1, type=float)
        parser.add_argument("--weight_decay", default=1e-2, type=float)

        parser.add_argument("--optimizer", default="sgd", type=str)

        parser.add_argument("--momentum", default=0.9, type=float)
        parser.add_argument("--beta1", default=0.9, type=float)
        parser.add_argument("--beta2", default=0.999, type=float)

        parser.add_argument("--eval_cutoff", default=100, type=int)

        return parser
