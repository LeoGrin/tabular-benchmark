# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, Optional, Union

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

import lib
from lib import KWArgs


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
    n_candidates: Union[None, int, float] = None
    eval_n_candidates: Union[None, int, float] = None
    freeze_contexts_after_n_epochs: Optional[int] = None

    def __post_init__(self):
        if self.freeze_contexts_after_n_epochs is not None:
            assert self.freeze_contexts_after_n_epochs >= 0


class Model(nn.Module):
    class ForwardOutput(NamedTuple):
        y_pred: Tensor
        context_idx: Tensor
        context_probs: Tensor

    def __init__(
        self,
        *,
        #
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
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
            nn.Linear(d_main, lib.get_d_out(n_classes)),
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

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
        x = torch.cat(x, dim=1)

        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    def forward(
        self,
        *,
        x_: dict[str, Tensor],
        y: Optional[Tensor],
        idx: Optional[Tensor],
        candidate_x_: dict[str, Tensor],
        candidate_y: Tensor,
        candidate_idx: Tensor,
        context_size: int,
        context_idx: Optional[Tensor],
        is_train: bool,
    ) -> ForwardOutput:
        # >>> E
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.memory_efficient
        ):
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
            assert y is not None
            assert idx is not None
            if context_idx is None:
                candidate_k = torch.cat([k, candidate_k])
                candidate_y = torch.cat([y, candidate_y])
                candidate_idx = torch.cat([idx, candidate_idx])
        else:
            assert y is None
            assert idx is None

        # >>>
        batch_size, d_main = k.shape
        device = k.device
        if context_idx is None:
            with torch.no_grad():
                if self.search_index is None:
                    self.search_index = (
                        faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                        if device.type == 'cuda'
                        else faiss.IndexFlatL2(d_main)
                    )
                self.search_index.reset()
                self.search_index.add(candidate_k)  # type: ignore[code]
                distances: Tensor
                distances, context_idx = self.search_index.search(  # type: ignore[code]
                    k, context_size + (1 if is_train else 0)
                )
                assert isinstance(context_idx, Tensor)
                if is_train:
                    distances[
                        context_idx == torch.arange(batch_size, device=device)[:, None]
                    ] = torch.inf
                    context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])
        # "absolute" means "not relative", i.e. the original indices in the train set.
        absolute_context_idx = candidate_idx[context_idx]

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
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

        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        raw_probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(raw_probs)

        context_y_emb: Tensor = self.label_encoder(candidate_y[context_idx][..., None])
        values: Tensor = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>>
        for block in self.blocks1:
            x: Tensor = x + block(x)
        x: Tensor = self.head(x)
        return Model.ForwardOutput(x, absolute_context_idx, raw_probs)


# NOTE: THIS EXPERIMENT IS NOT DESCRIBED IN THE PAPER.
# The CandidateQueue class is used when n{_eval}_candidates is not None in the config.
# This is an attempt to accelerate the training by using less then
# `train_size` candidates on each step.
# After quick testing, the context freeze technique looked more promising,
# so we abandoned the idea of subsampling.
# However, we may have missed something!
class CandidateQueue:
    def __init__(
        self, train_size: int, n_candidates: Union[int, float], device: torch.device
    ) -> None:
        assert train_size > 0
        if isinstance(n_candidates, int):
            assert 0 < n_candidates < train_size
            self._n_candidates = n_candidates
        else:
            assert 0.0 < n_candidates < 1.0
            self._n_candidates = int(n_candidates * train_size)
        self._train_size = train_size
        self._candidate_queue = torch.tensor([], dtype=torch.int64, device=device)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._candidate_queue) < self._n_candidates:
            self._candidate_queue = torch.cat(
                [
                    self._candidate_queue,
                    torch.randperm(
                        self._train_size, device=self._candidate_queue.device
                    ),
                ]
            )
        candidate_indices, self._candidate_queue = self._candidate_queue.split(
            [self._n_candidates, len(self._candidate_queue) - self._n_candidates]
        )
        return candidate_indices


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
    train_candidate_queue = (
        None
        if C.n_candidates is None
        else CandidateQueue(train_size, C.n_candidates, device)
    )
    eval_candidate_queue = (
        None
        if C.eval_n_candidates is None
        else CandidateQueue(train_size, C.eval_n_candidates, device)
    )
    # frozen_contexts are the fixed contexts for the "train" part.
    # They can be use to speed up the training.
    frozen_contexts: Optional[Tensor] = None
    # NOTE: uncomment this and other related lines to save a log of validation
    # contexts for analysis.
    # val_context_idx = []
    # val_context_probs = []

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

        is_train = part == 'train'
        if training and frozen_contexts is not None:
            candidate_indices, context_idx = frozen_contexts[idx].unique(
                return_inverse=True
            )
        else:
            # Importantly, `training`, not `is_train` should be used to choose the queue
            candidate_queue = (
                train_candidate_queue if training else eval_candidate_queue
            )
            candidate_indices = (
                train_indices if candidate_queue is None else next(candidate_queue)
            )
            context_idx = None
            if is_train:
                # This is not done when there are frozen contexts, because they are
                # already valid.
                candidate_indices = candidate_indices[
                    ~torch.isin(candidate_indices, idx)
                ]
        candidate_x, candidate_y = get_Xy(
            'train',
            None if candidate_indices is train_indices else candidate_indices,
        )

        fwd_out: Model.ForwardOutput = model(
            x_=x,
            y=y if is_train else None,
            idx=idx if is_train else None,
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            candidate_idx=candidate_indices,
            context_idx=context_idx,
            context_size=C.context_size,
            is_train=is_train,
        )
        return fwd_out._replace(y_pred=fwd_out.y_pred.squeeze(-1))

    @torch.inference_mode()
    def evaluate(parts: list[str], eval_batch_size: int, *, progress_bar: bool = False):
        model.eval()
        predictions = {}
        context_idx = {}
        context_probs = {}
        for part in parts:
            while eval_batch_size:
                try:
                    fwd_out = delu.cat(
                        [
                            apply_model(part, idx, False)
                            for idx in tqdm(
                                torch.arange(dataset.size(part), device=device).split(
                                    eval_batch_size
                                ),
                                desc=f'Evaluation ({part})',
                                disable=not progress_bar,
                            )
                        ]
                    )
                    predictions[part], context_idx[part], context_probs[part] = (
                        x.cpu().numpy() for x in fwd_out
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
        return metrics, predictions, context_idx, context_probs, eval_batch_size

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

        if (
            C.freeze_contexts_after_n_epochs is not None
            and epoch == C.freeze_contexts_after_n_epochs
        ):
            logger.info('Freezing contexts')
            _, _, context_idx, _, _ = evaluate(
                ['train'], eval_batch_size, progress_bar=True
            )
            frozen_contexts = torch.tensor(context_idx['train'], device=device)

        model.train()
        epoch_losses = []
        for batch_idx in tqdm(
            lib.make_random_batches(train_size, C.batch_size, device),
            desc=f'Epoch {epoch}',
        ):
            loss, new_chunk_size = lib.train_step(
                optimizer,
                lambda idx: loss_fn(apply_model('train', idx, True)[0], Y_train[idx]),
                batch_idx,
                chunk_size or C.batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or C.batch_size):
                chunk_size = new_chunk_size
                logger.warning(f'chunk_size = {chunk_size}')

        epoch_losses, mean_loss = lib.process_epoch_losses(epoch_losses)
        metrics, predictions, context_idx, context_probs, eval_batch_size = evaluate(
            ['val', 'test'], eval_batch_size
        )
        del context_idx, context_probs
        # val_context_idx.append(context_idx['val'])
        # val_context_probs.append(context_probs['val'])
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
    report['metrics'], predictions, *_ = evaluate(
        ['train', 'val', 'test'], eval_batch_size
    )
    report['chunk_size'] = chunk_size
    report['eval_batch_size'] = eval_batch_size
    lib.dump_predictions(predictions, output)
    # np.save(
    #     output / 'val_context_idx.npy',
    #     np.stack(val_context_idx, axis=1).astype(np.int32),
    # )
    # np.save(
    #     output / 'val_context_probs.npy',
    #     np.stack(val_context_probs, axis=1).astype(np.float32),
    # )
    lib.dump_summary(lib.summarize(report), output)
    save_checkpoint()
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)
