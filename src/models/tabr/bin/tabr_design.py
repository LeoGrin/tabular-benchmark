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


class Model(nn.Module):
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
        memory_efficient: bool = False,
        # Design options.
        dot_product: bool,
        use_Q: bool,
        use_V: bool,
        use_labels: bool,
        use_T: bool,
        self_attention: bool,
        scale_similarities: bool,
    ) -> None:
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        if use_T:
            assert not use_Q  # Because of unclear semantics.
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
        self.Q = nn.Linear(d_main, d_main) if use_Q else None
        self.K = nn.Linear(d_main, d_main)
        self.V = nn.Linear(d_main, d_main) if use_V else None
        self.T = (
            nn.Sequential(
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main, bias=False),
            )
            if use_T
            else None
        )
        if use_labels:
            self.label_encoder = (
                nn.Linear(1, d_main)
                if n_classes is None
                else nn.Sequential(
                    nn.Embedding(n_classes, d_main),
                    delu.nn.Lambda(lambda x: x.squeeze(-2)),
                )
            )
            self.label_placeholder = (
                nn.parameter.Parameter(torch.empty(d_main)) if self_attention else None
            )
        else:
            self.label_encoder = None
            self.label_placeholder = None
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
        self.dot_product = dot_product
        self.self_attention = self_attention
        self.scale_similarities = scale_similarities

        self.search_index = None
        self.memory_efficient = memory_efficient
        self.reset_parameters()

    def reset_parameters(self):
        if self.label_encoder is not None:
            if self.label_placeholder is not None:
                nn.init.uniform_(self.label_placeholder, -1.0, 1.0)
            if isinstance(self.label_encoder, nn.Linear):
                bound = 1 / math.sqrt(2.0)
                nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]
                nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]
            else:
                assert isinstance(self.label_encoder[0], nn.Embedding)
                nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]

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
        # xn ~ "x normalized"
        xn = x if self.normalization is None else self.normalization(x)
        return x, xn

    def forward(
        self,
        x_: dict[str, Tensor],
        y: Optional[Tensor],
        candidate_x_: dict[str, Tensor],
        candidate_y: Tensor,
        *,
        context_size: int,
        is_train: bool,
    ) -> Tensor:
        # >>>
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.memory_efficient
        ):
            candidate_xn = self._encode(candidate_x_)[1]
            candidate_k = self.K(candidate_xn)
        x, xn = self._encode(x_)
        k: Tensor = self.K(xn)
        q: Tensor = k if self.Q is None else self.Q(xn)
        if is_train:
            assert y is not None
            candidate_xn = torch.cat([xn, candidate_xn])
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None

        # >>> R
        batch_size, d_main = k.shape
        device = k.device
        with torch.no_grad():
            if self.search_index is None:
                self.search_index = (
                    (
                        faiss.GpuIndexFlatIP
                        if self.dot_product
                        else faiss.GpuIndexFlatL2
                    )(faiss.StandardGpuResources(), d_main)
                    if x.device.type == 'cuda'
                    else (faiss.IndexFlatIP if self.dot_product else faiss.IndexFlatL2)(
                        d_main
                    )
                )
            self.search_index.reset()
            self.search_index.add(candidate_k)  # type: ignore[code]
            faiss_scores: Tensor  # dot products if self.dot_product else L2 distances
            context_idx: Tensor
            faiss_scores, context_idx = self.search_index.search(  # type: ignore[code]
                q, context_size + (1 if is_train else 0)
            )
            if is_train:
                faiss_scores[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                ] = (-torch.inf if self.dot_product else torch.inf)
                context_idx = context_idx.gather(
                    -1, faiss_scores.argsort(descending=self.dot_product)[..., :-1]
                )

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            context_xn = self._encode(
                {
                    ftype: torch.cat([x_[ftype], candidate_x_[ftype]])[
                        context_idx
                    ].flatten(0, 1)
                    for ftype in x_
                }
            )[1]
            context_k = self.K(context_xn).reshape(batch_size, context_size, -1)
        else:
            context_xn = candidate_xn[context_idx]
            context_k = candidate_k[context_idx]

        dot_products = (q[..., None, :] @ context_k.transpose(-1, -2)).squeeze(-2)
        similarities = (
            dot_products
            if self.dot_product
            else (
                -q.square().sum(-1, keepdim=True)
                + 2 * dot_products
                - context_k.square().sum(-1)
            )
        )
        if self.self_attention:
            self_similarities = (
                (q * k).sum(-1, keepdim=True)
                if self.dot_product
                else torch.zeros(
                    *similarities.shape[:-1], 1, device=similarities.device
                )
                if q is k
                else (q - k).square().sum(-1, keepdim=True)
            )
            similarities = torch.cat([self_similarities, similarities], dim=-1)
        if self.scale_similarities:
            similarities = similarities / math.sqrt(d_main)
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        values = None
        if self.V is not None:
            context_v = self.V(context_xn)
            if self.self_attention:
                context_v = torch.cat([self.V(xn)[:, None, :], context_v], dim=1)
            values = context_v if values is None else values + context_v
        if self.label_encoder is not None:
            context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
            if self.self_attention:
                assert self.label_placeholder is not None
                context_y_emb = torch.cat(
                    [self.label_placeholder.expand(batch_size, 1, -1), context_y_emb],
                    dim=1,
                )
            values = context_y_emb if values is None else values + context_y_emb
        if self.T is not None:
            assert q is k
            k_diff = k[:, None] - context_k
            if self.self_attention:
                k_diff = torch.cat(
                    [torch.zeros(batch_size, 1, d_main, device=device), k_diff], dim=1
                )
            values = self.T(k_diff) if values is None else values + self.T(k_diff)
        assert values is not None
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

    def apply_model(part, idx):
        is_train = part == 'train'
        x, y = get_Xy(part, idx)
        return model(
            x,
            y if is_train else None,
            *get_Xy(
                'train',
                train_indices[~torch.isin(train_indices, idx)] if is_train else None,
            ),
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
                                apply_model(part, idx)
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
                lambda idx: loss_fn(apply_model('train', idx), Y_train[idx]),
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
