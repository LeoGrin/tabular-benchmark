# The model.

# >>>
# if __name__ == '__main__':
#     import os
#     import sys

#     _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
#     os.environ['PROJECT_DIR'] = _project_dir
#     sys.path.append(_project_dir)
#     del _project_dir
# # <<<
import os
os.environ["PROJECT_DIR"] = "models/tabr"
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import delu
import faiss
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm
import skorch
import time

#import lib
#from lib import KWArgs
import models.tabr.lib as lib
from models.tabr.lib import KWArgs

import hashlib

def hash_row(row):
    return hashlib.sha256(row.tobytes()).hexdigest()

def hash_2d_tensor_by_rows(X):
    return np.apply_along_axis(hash_row, axis=1, arr=X)

@dataclass(frozen=True)
class Config:
    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]  # lib.data.build_dataset
    model: KWArgs  # Model
    context_size: int
    optimizer: KWArgs  # lib.deep.make_optimizer
    batch_size: int
    patience: Optional[int]
    n_epochs: Union[int, float]


class InputShapeSetterTabR(skorch.callbacks.Callback):
    # skorch callback to set input-specific parameters
    def __init__(self, regression=False, batch_size=None,
                 categorical_indicator=None, categories=None):
        self.categorical_indicator = categorical_indicator
        self.regression = regression
        self.batch_size = batch_size
        self.categories = categories

    def on_train_begin(self, net, X, y):
        # unpack X
        X = X["X"]
        print("categorical_indicator", self.categorical_indicator)
        if self.categorical_indicator is None:
            d_numerical = X.shape[1]
            categories = []
        else:
            d_numerical = X.shape[1] - sum(self.categorical_indicator)
            print("self.categorical_indicator", self.categorical_indicator)
            print("self.categories", self.categories)
            print("X", X)
            if self.categories is None:
                categories = list((X[:, self.categorical_indicator].max(0) + 1).astype(int))
                # assert that it's the same that compuring with np.unique
                cat_cardinalities = []
                for i in range(X.shape[1]):
                    if self.categorical_indicator[i]:
                        print(f"cat {i}, {np.unique(X[:, i], return_counts=True)}")
                        cat_cardinalities.append(len(np.unique(X[:, i])))
                assert np.all(np.array(categories) == np.array(cat_cardinalities)), f"categories {categories} != cat_cardinalities {cat_cardinalities}"
            else:
                categories = self.categories
        #binary features are those that have only 2 categories
        if self.categorical_indicator is None:
            categorical_indicator_ = np.zeros(X.shape[1], dtype=bool)
        else:
            categorical_indicator_ = self.categorical_indicator
        # construct binary indicator #TODO: check if this is correct
        categorical_indices = np.where(self.categorical_indicator)[0]
        bin_indicator = np.zeros(X.shape[1], dtype=bool)
        for i in range(len(categories)):
            if categories[i] == 2:
                bin_indicator[categorical_indices[i]] = True
        y_train = torch.from_numpy(y)
        if self.regression:
            y_train = y_train.float()
        else:
            y_train = y_train.long()
        y_train = y_train.reshape(-1)
        X_train = torch.from_numpy(X).float()
        X_train_hashes = hash_2d_tensor_by_rows(X_train)
        # move to device
        X_train = X_train.to(net.device)
        y_train = y_train.to(net.device)
        print(f"Computed {len(np.unique(X_train_hashes))} unique hashes for X_train of shape {X_train.shape}")
        net.set_params(module__n_num_features=d_numerical,
            module__n_bin_features=bin_indicator.sum(),
            module__cat_cardinalities=categories, #FIXME #lib.get_categories(X_cat),
            module__n_classes=2 if self.regression == False else None, #FIXME
            module__categorical_indicator=categorical_indicator_,
            module__binary_indicator=bin_indicator,
            # save X_train and y_train to be able to use them during retrieval
            module__X_train=X_train,
            module__y_train=y_train,
            module__X_train_hashes=X_train_hashes,
            module__is_train=True,
        )
        print("Numerical features: {}".format(d_numerical))
        print("Categories {}".format(categories))
    def on_train_end(self, net, X, y):
        net.set_params(module__is_train=False)


class Model(nn.Module):
    def __init__(
        self,
        *,
        #
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
        categorical_indicator=None,  #MODIF
        binary_indicator=None,  #MODIF

        is_train: bool,  #MODIF

        X_train=None, #MODIF
        y_train=None,  #MODIF
        X_train_hashes=None,  #MODIF

        #
        num_embeddings: Optional[dict],  # lib.deep.ModuleSpec
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization: Union[bool, Literal['auto']],
        context_dropout: float,
        dropout0: float,
        dropout1: Union[float, Literal['dropout0']],
        normalization: str,
        activation: str,
        #
        # The following options should be used only when truly needed.
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
    ) -> None:
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        super().__init__()
        if dropout1 == 'dropout0':
            dropout1 = dropout0
        self.one_hot_encoder = (
            lib.OneHotEncoder(cat_cardinalities) if cat_cardinalities else None
        )
        self.num_embeddings = (
            None
            if num_embeddings is None
            else lib.make_module(num_embeddings, n_features=n_num_features)
        )

        # >>> E
        d_in = (
            n_num_features
            * (1 if num_embeddings is None else num_embeddings['d_embedding'])
            + n_bin_features
            + sum(cat_cardinalities)
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)]
        )

        # >>> R
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes is None
            else nn.Sequential(
                nn.Embedding(n_classes, d_main), delu.nn.Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> P
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(),
            nn.Linear(d_main, lib.get_d_out(n_classes) if n_classes != 2 else 2), #MODIF: we always use cross entropy loss
        )
        print("n_classes", n_classes)

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

        #MODIF
        self.categorical_indicator = categorical_indicator
        self.binary_indicator = binary_indicator
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_hashes = X_train_hashes
        self.is_train = is_train


    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]  # noqa: E501
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501

    def _encode(self, x_: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x_num = x_.get('num')
        x_bin = x_.get('bin')
        x_cat = x_.get('cat')
        del x_

        x = []
        if x_num is None:
            assert self.num_embeddings is None
        else:
            x.append(
                x_num
                if self.num_embeddings is None
                else self.num_embeddings(x_num).flatten(1)
            )
        if x_bin is not None:
            x.append(x_bin)
        if x_cat is None:
            assert self.one_hot_encoder is None
        else:
            assert self.one_hot_encoder is not None
            x.append(self.one_hot_encoder(x_cat))
        assert x
        x = torch.cat(x, dim=1).float()

        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k
    
    #MODIF
    def remove_train_batch_from_candidate(self, train_batch):
        # having access to the batch indices is a pain in skorch, so we hack around it
        # by remove rows with the same hash as the train batch
        start_time = time.time()
        candidate_idx = np.arange(self.X_train.shape[0])
        candidates_hashes = self.X_train_hashes
        train_batch_hashes = hash_2d_tensor_by_rows(train_batch.cpu())
        # Create a dictionary where keys are hashes and values are counts of their occurrences in train_batch_hashes
        hash_counts = {hash: np.count_nonzero(train_batch_hashes == hash) for hash in np.unique(train_batch_hashes)}

        assert np.isin(train_batch_hashes, candidates_hashes).all(), "train_batch_hashes should be a subset of candidates_hashes"

        # Iterate over candidate_hashes and create a mask. Decrement the count in hash_counts every time a hash is encountered.
        mask = []
        for hash in candidates_hashes:
            if hash in hash_counts and hash_counts[hash] > 0:
                mask.append(False)  # This will not select the index
                hash_counts[hash] -= 1
            else:
                mask.append(True)  # This will select the index

        # Select the desired indices
        selected_indices = candidate_idx[mask]

        #print(f"remove_train_batch_from_candidate took {time.time() - start_time} seconds")

        return selected_indices

    
    def forward(
        self,
        #*, #MODIF
        #x_: dict[str, Tensor],
        X: Tensor, #MODIF
        y: Optional[Tensor]=None, #MODIF
        #candidate_x_: dict[str, Tensor],
        #candidate_y: Tensor,
        #context_size: int,
        #is_train: bool,
    ) -> Tensor:
        # >>>
        #MODIF
        # reshape y
        if y is not None:
            y = y.reshape(-1)
        # transform x_ to dict
        x_ = {}
        assert self.categorical_indicator is not None, "categorical_indicator must be set"
        assert self.binary_indicator is not None, "binary_indicator must be set"
        x_["num"] = X[:, ~self.categorical_indicator & ~self.binary_indicator]
        x_["bin"] = X[:, self.binary_indicator].long()
        x_["cat"] = X[:, self.categorical_indicator].long()
        for t in ["num", "bin", "cat"]:
            if x_[t].shape[1] == 0:
                x_[t] = None

        #TODO: remove current batch from candidates
        if self.is_train:
            candidate_indices = self.remove_train_batch_from_candidate(X)
        else:
            candidate_indices = np.arange(self.X_train.shape[0])
        candidate_x_ = {}
        candidate_x_["num"] = self.X_train[candidate_indices][:, ~self.categorical_indicator & ~self.binary_indicator]
        candidate_x_["bin"] = self.X_train[candidate_indices][:, self.binary_indicator].long()
        candidate_x_["cat"] = self.X_train[candidate_indices][:, self.categorical_indicator].long()
        for t in ["num", "bin", "cat"]:
            if candidate_x_[t].shape[1] == 0:
                candidate_x_[t] = None

        candidate_y = self.y_train[candidate_indices]

        context_size = 96 #TODO: check

        is_train = self.is_train


        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.memory_efficient
        ):
            # NOTE: during evaluation, candidate keys can be computed just once, which
            # looks like an easy opportunity for optimization. However:
            # - if your dataset is small or/and the encoder is just a linear layer
            #   (no embeddings and encoder_n_blocks=0), then encoding candidates
            #   is not a bottleneck.
            # - implementing this optimization makes the code complex and/or unobvious,
            #   because there are many things that should be taken into account:
            #     - is the input coming from the "train" part?
            #     - is self.training True or False?
            #     - is PyTorch autograd enabled?
            #     - is saving and loading checkpoints handled correctly?
            # This is why we do not implement this optimization.

            # When memory_efficient is True, this potentially heavy computation is
            # performed without gradients.
            # Later, it is recomputed with gradients only for the context objects.
            candidate_k = (
                self._encode(candidate_x_)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(x)[1]
                        for x in delu.iter_batches(
                            candidate_x_, self.candidate_encoding_batch_size
                        )
                    ]
                )
            )
        x, k = self._encode(x_)
        if is_train:
            # NOTE: here, we add the training batch back to the candidates after the
            # function `apply_model` removed them. The further code relies
            # on the fact that the first batch_size candidates come from the
            # training batch.
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None

        # >>>
        # The search below is optimized for larger datasets and is significantly faster
        # than the naive solution (keep autograd on + manually compute all pairwise
        # squared L2 distances + torch.topk).
        # For smaller datasets, however, the naive solution can actually be faster.
        batch_size, d_main = k.shape
        device = k.device
        with torch.no_grad():
            if self.search_index is None:
                self.search_index = (
                    faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                    if device.type == 'cuda'
                    else faiss.IndexFlatL2(d_main)
                )
            # Updating the index is much faster than creating a new one.
            self.search_index.reset()
            self.search_index.add(candidate_k)  # type: ignore[code]
            distances: Tensor
            context_idx: Tensor
            distances, context_idx = self.search_index.search(  # type: ignore[code]
                k, context_size + (1 if is_train else 0)
            )
            if is_train:
                # NOTE: to avoid leakage, the index i must be removed from the i-th row,
                # (because of how candidate_k is constructed).
                distances[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                ] = torch.inf
                # Not the most elegant solution to remove the argmax, but anyway.
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            # Repeating the same computation,
            # but now only for the context objects and with autograd on.
            context_k = self._encode(
                {
                    ftype: torch.cat([x_[ftype], candidate_x_[ftype]])[
                        context_idx
                    ].flatten(0, 1)
                    for ftype in x_
                }
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]

        # In theory, when autograd is off, the distances obtained during the search
        # can be reused. However, this is not a bottleneck, so let's keep it simple
        # and use the same code to compute `similarities` during both
        # training and evaluation.
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)
        context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>>
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x


def main(
    config: lib.JSONDict, output: Union[str, Path], *, force: bool = False
) -> Optional[lib.JSONDict]:
    # >>> start
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    delu.random.seed(C.seed)
    device = lib.get_device()

    # >>> data
    dataset = (
        C.data if isinstance(C.data, lib.Dataset) else lib.build_dataset(**C.data)
    ).to_torch(device)
    if dataset.is_regression:
        dataset.data['Y'] = {k: v.float() for k, v in dataset.Y.items()}
    Y_train = dataset.Y['train'].to(
        torch.long if dataset.is_multiclass else torch.float
    )

    # >>> model
    model = Model(
        n_num_features=dataset.n_num_features,
        n_bin_features=dataset.n_bin_features,
        cat_cardinalities=dataset.cat_cardinalities(),
        n_classes=dataset.n_classes(),
        **C.model,
    )
    report['n_parameters'] = lib.get_n_parameters(model)
    logger.info(f'n_parameters = {report["n_parameters"]}')
    report['prediction_type'] = None if dataset.is_regression else 'logits'
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # type: ignore[code]

    # >>> training
    def zero_wd_condition(
        module_name: str,
        module: nn.Module,
        parameter_name: str,
        parameter: nn.parameter.Parameter,
    ):
        return (
            'label_encoder' in module_name
            or 'label_encoder' in parameter_name
            or lib.default_zero_weight_decay_condition(
                module_name, module, parameter_name, parameter
            )
        )

    optimizer = lib.make_optimizer(
        model, **C.optimizer, zero_weight_decay_condition=zero_wd_condition
    )
    loss_fn = lib.get_loss_fn(dataset.task_type)

    train_size = dataset.size('train')
    train_indices = torch.arange(train_size, device=device)

    epoch = 0
    eval_batch_size = 32768
    chunk_size = None
    progress = delu.ProgressTracker(C.patience)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]

    def get_Xy(part: str, idx) -> tuple[dict[str, Tensor], Tensor]:
        batch = (
            {
                key[2:]: dataset.data[key][part]
                for key in dataset.data
                if key.startswith('X_')
            },
            dataset.Y[part],
        )
        return (
            batch
            if idx is None
            else ({k: v[idx] for k, v in batch[0].items()}, batch[1][idx])
        )

    def apply_model(part: str, idx: Tensor, training: bool):
        x, y = get_Xy(part, idx)

        candidate_indices = train_indices
        is_train = part == 'train'
        if is_train:
            # NOTE: here, the training batch is removed from the candidates.
            # It will be added back inside the model's forward pass.
            candidate_indices = candidate_indices[~torch.isin(candidate_indices, idx)]
        candidate_x, candidate_y = get_Xy(
            'train',
            # This condition is here for historical reasons, it could be just
            # the unconditional `candidate_indices`.
            None if candidate_indices is train_indices else candidate_indices,
        )

        return model(
            x_=x,
            y=y if is_train else None,
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            context_size=C.context_size,
            is_train=is_train,
        ).squeeze(-1)

    @torch.inference_mode()
    def evaluate(parts: list[str], eval_batch_size: int):
        model.eval()
        predictions = {}
        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx, False)
                                for idx in torch.arange(
                                    dataset.size(part), device=device
                                ).split(eval_batch_size)
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    logger.warning(f'eval_batch_size = {eval_batch_size}')
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
        metrics = (
            dataset.calculate_metrics(predictions, report['prediction_type'])
            if lib.are_valid_predictions(predictions)
            else {x: {'score': -999999.0} for x in predictions}
        )
        return metrics, predictions, eval_batch_size

    def save_checkpoint():
        lib.dump_checkpoint(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'random_state': delu.random.get_state(),
                'progress': progress,
                'report': report,
                'timer': timer,
                'training_log': training_log,
            },
            output,
        )
        lib.dump_report(report, output)
        lib.backup_output(output)

    print()
    timer = lib.run_timer()
    while epoch < C.n_epochs:
        print(f'[...] {lib.try_get_relative_path(output)} | {timer}')

        model.train()
        epoch_losses = []
        for batch_idx in tqdm(
            lib.make_random_batches(train_size, C.batch_size, device),
            desc=f'Epoch {epoch}',
        ):
            loss, new_chunk_size = lib.train_step(
                optimizer,
                lambda idx: loss_fn(apply_model('train', idx, True), Y_train[idx]),
                batch_idx,
                chunk_size or C.batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or C.batch_size):
                chunk_size = new_chunk_size
                logger.warning(f'chunk_size = {chunk_size}')

        epoch_losses, mean_loss = lib.process_epoch_losses(epoch_losses)
        metrics, predictions, eval_batch_size = evaluate(
            ['val', 'test'], eval_batch_size
        )
        lib.print_metrics(mean_loss, metrics)
        training_log.append(
            {'epoch-losses': epoch_losses, 'metrics': metrics, 'time': timer()}
        )
        writer.add_scalars('loss', {'train': mean_loss}, epoch, timer())
        for part in metrics:
            writer.add_scalars('score', {part: metrics[part]['score']}, epoch, timer())

        progress.update(metrics['val']['score'])
        if progress.success:
            lib.celebrate()
            report['best_epoch'] = epoch
            report['metrics'] = metrics
            save_checkpoint()
            lib.dump_predictions(predictions, output)

        elif progress.fail or not lib.are_valid_predictions(predictions):
            break

        epoch += 1
        print()
    report['time'] = str(timer)

    # >>> finish
    model.load_state_dict(lib.load_checkpoint(output)['model'])
    report['metrics'], predictions, _ = evaluate(
        ['train', 'val', 'test'], eval_batch_size
    )
    report['chunk_size'] = chunk_size
    report['eval_batch_size'] = eval_batch_size
    lib.dump_predictions(predictions, output)
    lib.dump_summary(lib.summarize(report), output)
    save_checkpoint()
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)
