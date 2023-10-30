import math
import statistics
from functools import partial
from typing import Any, Callable, Optional, Union, cast

import delu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn.parameter import Parameter

from .util import TaskType, is_oom_exception

# ======================================================================================
# >>> modules <<<
# ======================================================================================
# When an instance of ModuleSpec is a dict,
# it must contain the key "type" with a string value
ModuleSpec = Union[str, dict[str, Any], Callable[..., nn.Module]]


def _initialize_embeddings(weight: Tensor, d: Optional[int]) -> None:
    if d is None:
        d = weight.shape[-1]
    d_sqrt_inv = 1 / math.sqrt(d)
    nn.init.uniform_(weight, a=-d_sqrt_inv, b=d_sqrt_inv)


def make_trainable_vector(d: int) -> Parameter:
    x = torch.empty(d)
    _initialize_embeddings(x, None)
    return Parameter(x)


class OneHotEncoder(nn.Module):
    cardinalities: Tensor

    def __init__(self, cardinalities: list[int]) -> None:
        # cardinalities[i]`` is the number of unique values for the i-th categorical feature.
        super().__init__()
        self.register_buffer('cardinalities', torch.tensor(cardinalities))

    def forward(self, x: Tensor) -> Tensor:
        encoded_columns = [
            F.one_hot(x[..., column], cardinality)
            for column, cardinality in zip(range(x.shape[-1]), self.cardinalities)
        ]

        return torch.cat(encoded_columns, -1)


class CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.weight = make_trainable_vector(d_embedding)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3
        assert x.shape[-1] == len(self.weight)
        return torch.cat([self.weight.expand(len(x), 1, -1), x], dim=1)


class CatEmbeddings(nn.Module):
    def __init__(
        self,
        _cardinalities_and_maybe_dimensions: Union[list[int], list[tuple[int, int]]],
        d_embedding: Optional[int] = None,
        *,
        stack: bool = False,
    ) -> None:
        assert _cardinalities_and_maybe_dimensions
        spec = _cardinalities_and_maybe_dimensions
        if not (
            (isinstance(spec[0], tuple) and d_embedding is None)
            or (isinstance(spec[0], int) and d_embedding is not None)
        ):
            raise ValueError(
                'Invalid arguments. Valid combinations are:'
                ' (1) the first argument is a list of (cardinality, embedding)-tuples AND d_embedding is None'
                ' (2) the first argument is a list of cardinalities AND d_embedding is an integer'
            )
        if stack and d_embedding is None:
            raise ValueError('stack can be True only when d_embedding is not None')

        super().__init__()
        spec_ = cast(
            list[tuple[int, int]],
            spec if d_embedding is None else [(x, d_embedding) for x in spec],
        )
        self._embeddings = nn.ModuleList()
        for cardinality, d_embedding in spec_:
            self._embeddings.append(nn.Embedding(cardinality, d_embedding))
        self.stack = stack
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self._embeddings:
            _initialize_embeddings(module.weight, None)  # type: ignore[code]

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        assert x.shape[1] == len(self._embeddings)
        out = [module(column) for module, column in zip(self._embeddings, x.T)]
        return torch.stack(out, dim=1) if self.stack else torch.cat(out, dim=1)


class LinearEmbeddings(nn.Module):
    def __init__(self, n_features: int, d_embedding: int, bias: bool = True):
        super().__init__()
        self.weight = Parameter(Tensor(n_features, d_embedding))
        self.bias = Parameter(Tensor(n_features, d_embedding)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                _initialize_embeddings(parameter, parameter.shape[-1])

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class PeriodicEmbeddings(nn.Module):
    def __init__(
        self, n_features: int, n_frequencies: int, frequency_scale: float
    ) -> None:
        super().__init__()
        self.frequencies = Parameter(
            torch.normal(0.0, frequency_scale, (n_features, n_frequencies))
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x


class NLinear(nn.Module):
    def __init__(
        self, n_features: int, d_in: int, d_out: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(Tensor(n_features, d_in, d_out))
        self.bias = Parameter(Tensor(n_features, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n_features):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class LREmbeddings(nn.Sequential):
    """The LR embeddings from the paper 'On Embeddings for Numerical Features in Tabular Deep Learning'."""  # noqa: E501

    def __init__(self, n_features: int, d_embedding: int) -> None:
        super().__init__(LinearEmbeddings(n_features, d_embedding), nn.ReLU())


class PLREmbeddings(nn.Sequential):
    """The PLR embeddings from the paper 'On Embeddings for Numerical Features in Tabular Deep Learning'.

    Additionally, the 'lite' option is added. Setting it to `False` gives you the original PLR
    embedding from the above paper. We noticed that `lite=True` makes the embeddings
    noticeably more lightweight without critical performance loss, and we used that for our model.
    """  # noqa: E501

    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
        lite: bool,
    ) -> None:
        super().__init__(
            PeriodicEmbeddings(n_features, n_frequencies, frequency_scale),
            (
                nn.Linear(2 * n_frequencies, d_embedding)
                if lite
                else NLinear(n_features, 2 * n_frequencies, d_embedding)
            ),
            nn.ReLU(),
        )


class MLP(nn.Module):
    class Block(nn.Module):
        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: str,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = make_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    Head = nn.Linear

    def __init__(
        self,
        *,
        d_in: int,
        d_out: Optional[int],
        n_blocks: int,
        d_layer: int,
        activation: str,
        dropout: float,
    ) -> None:
        assert n_blocks > 0
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                MLP.Block(
                    d_in=d_layer if block_i else d_in,
                    d_out=d_layer,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for block_i in range(n_blocks)
            ]
        )
        self.head = None if d_out is None else MLP.Head(d_layer, d_out)

    @property
    def d_out(self) -> int:
        return (
            self.blocks[-1].linear.out_features  # type: ignore[code]
            if self.head is None
            else self.head.out_features
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        if self.head is not None:
            x = self.head(x)
        return x


_CUSTOM_MODULES = {
    x.__name__: x
    for x in [
        LinearEmbeddings,
        LREmbeddings,
        PLREmbeddings,
        MLP,
    ]
}


def register_module(key: str, f: Callable[..., nn.Module]) -> None:
    assert key not in _CUSTOM_MODULES
    _CUSTOM_MODULES[key] = f


def make_module(spec: ModuleSpec, *args, **kwargs) -> nn.Module:
    """
    >>> make_module('ReLU')
    >>> make_module(nn.ReLU)
    >>> make_module('Linear', 1, out_features=2)
    >>> make_module((lambda *args: nn.Linear(*args)), 1, out_features=2)
    >>> make_module({'type': 'Linear', 'in_features' 1}, out_features=2)
    """
    if isinstance(spec, str):
        Module = getattr(nn, spec, None)
        if Module is None:
            Module = _CUSTOM_MODULES[spec]
        else:
            assert spec not in _CUSTOM_MODULES
        return make_module(Module, *args, **kwargs)
    elif isinstance(spec, dict):
        assert not (set(spec) & set(kwargs))
        spec = spec.copy()
        return make_module(spec.pop('type'), *args, **spec, **kwargs)
    elif callable(spec):
        return spec(*args, **kwargs)
    else:
        raise ValueError()


def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)


def get_d_out(n_classes: Optional[int]) -> int:
    return 1 if n_classes is None or n_classes == 2 else n_classes


# ======================================================================================
# >>> optimization <<<
# ======================================================================================
def default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: Parameter
):
    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.LayerNorm,
            nn.InstanceNorm1d,
            LinearEmbeddings,
            PeriodicEmbeddings,
        ),
    )


def make_parameter_groups(
    model: nn.Module,
    zero_weight_decay_condition,
    custom_groups: dict[tuple[str], dict],  # [(fullnames, options), ...]
) -> list[dict[str, Any]]:
    custom_fullnames = set()
    custom_fullnames.update(*custom_groups)
    assert sum(map(len, custom_groups)) == len(
        custom_fullnames
    ), 'Custom parameter groups must not intersect'

    parameters_info = {}  # fullname -> (parameter, needs_wd)
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters():
            fullname = f'{module_name}.{name}' if module_name else name
            parameters_info.setdefault(fullname, (parameter, []))[1].append(
                not zero_weight_decay_condition(module_name, module, name, parameter)
            )
    parameters_info = {k: (v[0], all(v[1])) for k, v in parameters_info.items()}

    params_with_wd = {'params': []}
    params_without_wd = {'params': [], 'weight_decay': 0.0}
    custom_params = {k: {'params': []} | v for k, v in custom_groups.items()}

    for fullname, (parameter, needs_wd) in parameters_info.items():
        for fullnames, group in custom_params.items():
            if fullname in fullnames:
                custom_fullnames.remove(fullname)
                group['params'].append(parameter)
                break
        else:
            (params_with_wd if needs_wd else params_with_wd)['params'].append(parameter)
    assert (
        not custom_fullnames
    ), f'Some of the custom parameters were not found in the model: {custom_fullnames}'
    return [params_with_wd, params_without_wd] + list(custom_params.values())


def make_optimizer(
    module: nn.Module,
    type: str,
    *,
    zero_weight_decay_condition=default_zero_weight_decay_condition,
    custom_parameter_groups: Optional[dict[tuple[str], dict]] = None,
    **optimizer_kwargs,
) -> torch.optim.Optimizer:
    if custom_parameter_groups is None:
        custom_parameter_groups = {}
    Optimizer = getattr(optim, type)
    parameter_groups = make_parameter_groups(
        module, zero_weight_decay_condition, custom_parameter_groups
    )
    return Optimizer(parameter_groups, **optimizer_kwargs)


def get_lr(optimizer: optim.Optimizer) -> float:
    return next(iter(optimizer.param_groups))['lr']


def set_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group['lr'] = lr


# ======================================================================================
# >>> training <<<
# ======================================================================================
def get_loss_fn(task_type: TaskType, **kwargs) -> Callable[..., Tensor]:
    loss_fn = (
        F.binary_cross_entropy_with_logits
        if task_type == TaskType.BINCLASS
        else F.cross_entropy
        if task_type == TaskType.MULTICLASS
        else F.mse_loss
    )
    return partial(loss_fn, **kwargs) if kwargs else loss_fn


def make_random_batches(
    train_size: int, batch_size: int, device: Optional[torch.device] = None
) -> list[Tensor]:
    permutation = torch.randperm(train_size, device=device)
    batches = permutation.split(batch_size)
    # Below, we check that we do not face this issue:
    # https://github.com/pytorch/vision/issues/3816
    # This is still noticeably faster than running randperm on CPU.
    # UPDATE: after thousands of experiments, we faced the issue zero times,
    # so maybe we should remove the assert.
    assert torch.equal(
        torch.arange(train_size, device=device), permutation.sort().values
    )
    return batches  # type: ignore[code]


def train_step(
    optimizer: optim.Optimizer,
    step_fn: Callable[..., Tensor],
    batch,
    chunk_size: int,
) -> tuple[Tensor, int]:
    """The standard training step.

    Additionally, when the step for the whole batch does not fit into GPU,
    this function automatically splits the batch into chunks (virtual batches).
    Note that this does not affect the algorithm.
    """
    batch_size = len(batch)
    random_state = delu.random.get_state()
    loss = None
    while chunk_size != 0:
        try:
            delu.random.set_state(random_state)
            optimizer.zero_grad()
            if batch_size <= chunk_size:
                loss = step_fn(batch)
                loss.backward()
            else:
                loss = None
                for chunk in delu.iter_batches(batch, chunk_size):
                    chunk_loss = step_fn(chunk)
                    chunk_loss = chunk_loss * (len(chunk) / batch_size)
                    chunk_loss.backward()
                    if loss is None:
                        loss = chunk_loss.detach()
                    else:
                        loss += chunk_loss.detach()
        except RuntimeError as err:
            if not is_oom_exception(err):
                raise
            delu.hardware.free_memory()
            chunk_size //= 2
        else:
            break
    if not chunk_size:
        raise RuntimeError('Not enough memory even for batch_size=1')
    optimizer.step()
    return cast(Tensor, loss), chunk_size


def process_epoch_losses(losses: list[Tensor]) -> tuple[list[float], float]:
    losses_ = torch.stack(losses).tolist()
    return losses_, statistics.mean(losses_)
