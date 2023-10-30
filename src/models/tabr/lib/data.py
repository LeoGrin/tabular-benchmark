import enum
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Generic, Iterable, Optional, TypeVar, Union, cast

import numpy as np
import sklearn.preprocessing
import torch
from loguru import logger
from torch import Tensor

from . import env, util
from .metrics import PredictionType
from .metrics import calculate_metrics as calculate_metrics_
from .util import Part, TaskType

NumpyDict = dict[str, np.ndarray]

_SCORE_SHOULD_BE_MAXIMIZED = {
    'accuracy': True,
    'cross-entropy': False,
    'mae': False,
    'r2': True,
    'rmse': False,
    'roc-auc': True,
}


class NumPolicy(enum.Enum):
    STANDARD = 'standard'
    QUANTILE = 'quantile'


class CatPolicy(enum.Enum):
    ORDINAL = 'ordinal'
    ONE_HOT = 'one-hot'


class YPolicy(enum.Enum):
    STANDARD = 'standard'


@dataclass
class YInfo:
    mean: float
    std: float


T = TypeVar('T', np.ndarray, Tensor)


def _to_numpy(x: Union[np.ndarray, Tensor]) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else x.cpu().numpy()


@dataclass(frozen=True)
class Dataset(Generic[T]):
    data: dict[str, dict[str, T]]  # {type: {part: <data>}}
    task_type: TaskType
    score: str
    y_info: Optional[YInfo]
    _Y_numpy: Optional[NumpyDict]  # this is used in calculate_metrics

    def __post_init__(self):
        if self.y_info is not None:
            assert self.is_regression
        for key in ['X_num', 'X_bin']:
            if key in self.data:
                assert all(
                    not (
                        np.isnan(x).any()
                        if isinstance(x, np.ndarray)
                        else x.isnan().any().cpu().item()
                    )
                    for x in self.data[key].values()
                )

    @classmethod
    def from_dir(
        cls, path: Union[Path, str], score: Optional[str]
    ) -> 'Dataset[np.ndarray]':
        path = env.get_path(path)
        info = util.load_json(path / 'info.json')
        task_type = TaskType(info['task_type'])
        if score is None:
            score = {
                TaskType.BINCLASS: 'accuracy',
                TaskType.MULTICLASS: 'accuracy',
                TaskType.REGRESSION: 'rmse',
            }[task_type]
        return Dataset(
            {
                key: {
                    part.value: np.load(
                        path / f'{key}_{part.value}.npy', allow_pickle=True
                    )
                    for part in Part
                }
                for key in ['X_num', 'X_bin', 'X_cat', 'Y']
                if path.joinpath(f'{key}_train.npy').exists()
            },
            task_type,
            score,
            None,
            None,
        )

    def _is_numpy(self) -> bool:
        return isinstance(
            next(iter(next(iter(self.data.values())).values())), np.ndarray
        )

    def _is_torch(self) -> bool:
        return not self._is_numpy()

    def to_torch(self, device=None) -> 'Dataset[Tensor]':
        if self._is_torch():
            return self  # type: ignore[code]
        data = {
            key: {
                part: torch.as_tensor(value).to(device)
                for part, value in self.data[key].items()
            }
            for key in self.data
        }
        return replace(self, data=data, _Y_numpy=self.data['Y'])  # type: ignore[code]

    def to_numpy(self) -> 'Dataset[np.ndarray]':
        if self._is_numpy():
            return self  # type: ignore[code]
        data = {
            key: {part: value.cpu().numpy() for part, value in self.data[key].items()}
            for key in self.data
        }
        return replace(self, data=data, _Y_numpy=None)  # type: ignore[code]

    @property
    def X_num(self) -> Optional[dict[str, T]]:
        return self.data.get('X_num')

    @property
    def X_bin(self) -> Optional[dict[str, T]]:
        return self.data.get('X_bin')

    @property
    def X_cat(self) -> Optional[dict[str, T]]:
        return self.data.get('X_cat')

    @property
    def Y(self) -> dict[str, T]:
        return self.data['Y']

    def merge_num_bin(self) -> 'Dataset[T]':
        if self.X_bin is None:
            return self
        else:
            data = self.data.copy()
            X_bin = data.pop('X_bin')
            if self.X_num is None:
                data['X_num'] = X_bin
            else:
                assert self._is_numpy()
                data['X_num'] = {
                    k: np.concatenate([self.X_num[k], X_bin[k]], 1) for k in self.X_num
                }
        return replace(self, data=data)

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_classification(self) -> bool:
        return self.is_binclass or self.is_multiclass

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]

    @property
    def n_bin_features(self) -> int:
        return 0 if self.X_bin is None else self.X_bin['train'].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_bin_features + self.n_cat_features

    def parts(self) -> Iterable[str]:
        return iter(next(iter(self.data.values())))

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.Y.values())) if part is None else len(self.Y[part])

    def n_classes(self) -> Optional[int]:
        return (
            None
            if self.is_regression
            else len((np.unique if self._is_numpy() else torch.unique)(self.Y['train']))
        )

    def cat_cardinalities(self) -> list[int]:
        unique = np.unique if self._is_numpy() else torch.unique
        return (
            []
            if self.X_cat is None
            else [len(unique(column)) for column in self.X_cat['train'].T]
        )

    def calculate_metrics(
        self,
        predictions: dict[str, Union[np.ndarray, Tensor]],
        prediction_type: Union[None, str, PredictionType],
    ) -> dict[str, Any]:
        if self._is_numpy():
            Y_ = cast(NumpyDict, self.Y)
        elif self._Y_numpy is not None:
            Y_ = self._Y_numpy
        else:
            raise RuntimeError()
        metrics = {
            part: calculate_metrics_(
                Y_[part],
                _to_numpy(predictions[part]),
                self.task_type,
                prediction_type,
                None if self.y_info is None else self.y_info.std,
            )
            for part in predictions
        }
        for part_metrics in metrics.values():
            part_metrics['score'] = (
                1.0 if _SCORE_SHOULD_BE_MAXIMIZED[self.score] else -1.0
            ) * part_metrics[self.score]
        return metrics


# Inspired by: https://github.com/Yura52/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def transform_num(X: NumpyDict, policy: NumPolicy, seed: Optional[int]) -> NumpyDict:
    X_train = X['train']
    if policy == NumPolicy.STANDARD:
        normalizer = sklearn.preprocessing.StandardScaler()
    elif policy == NumPolicy.QUANTILE:
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=1_000_000_000,  # i.e. no subsampling
            random_state=seed,
        )
        noise = 1e-3
        if noise > 0:
            # Noise is added to get a bit nicer transformation
            # for features with few unique values.
            assert seed is not None
            stds = np.std(X_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
            X_train = X_train + noise_std * np.random.default_rng(seed).standard_normal(
                X_train.shape
            )
    else:
        raise ValueError('Unknown normalization: ' + policy)
    normalizer.fit(X_train)
    return {k: normalizer.transform(v) for k, v in X.items()}  # type: ignore[code]


def transform_cat(X: NumpyDict, policy: CatPolicy) -> NumpyDict:
    # >>> ordinal encoding (even if encoding == .ONE_HOT)
    unknown_value = np.iinfo('int64').max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',  # type: ignore[code]
        unknown_value=unknown_value,  # type: ignore[code]
        dtype='int64',  # type: ignore[code]
    ).fit(X['train'])
    X = {k: encoder.transform(v) for k, v in X.items()}
    max_values = X['train'].max(axis=0)
    for part in ['val', 'test']:
        for column_idx in range(X[part].shape[1]):
            X[part][X[part][:, column_idx] == unknown_value, column_idx] = (
                max_values[column_idx] + 1
            )

    # >>> encode
    if policy == CatPolicy.ORDINAL:
        return X
    elif policy == CatPolicy.ONE_HOT:
        encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse=False, dtype=np.float32  # type: ignore[code]
        )
        encoder.fit(X['train'])
        return {k: cast(np.ndarray, encoder.transform(v)) for k, v in X.items()}
    else:
        raise ValueError(f'Unknown encoding: {policy}')


def transform_y(Y: NumpyDict, policy: YPolicy) -> tuple[NumpyDict, YInfo]:
    if policy == YPolicy.STANDARD:
        mean, std = float(Y['train'].mean()), float(Y['train'].std())
        Y = {k: (v - mean) / std for k, v in Y.items()}
        return Y, YInfo(mean, std)
    else:
        raise ValueError('Unknown policy: ' + policy)


def build_dataset(
    *,
    path: Union[str, Path],
    num_policy: Union[None, str, NumPolicy],
    cat_policy: Union[None, str, CatPolicy],
    y_policy: Union[None, str, YPolicy],
    score: Optional[str] = None,
    seed: int,
    cache: bool,
) -> Dataset[np.ndarray]:
    path = env.get_path(path)
    if cache:
        args = locals()
        args.pop('cache')
        args.pop('path')
        cache_path = (
            env.CACHE_DIR
            / f'build_dataset__{path.name}__{"__".join(map(str, args.values()))}__{hashlib.md5(str(args).encode("utf-8")).hexdigest()}.pickle'
        )
        if cache_path.exists():
            cached_args, cached_value = util.load_pickle(cache_path)
            if args == cached_args:
                logger.info(f"Using cached dataset: {cache_path.name}")
                return cached_value
            else:
                raise RuntimeError(f'Hash collision for {cache_path}')
    else:
        args = None
        cache_path = None

    logger.info(f"Building dataset (path: {path})")
    if num_policy is not None:
        num_policy = NumPolicy(num_policy)
    if cat_policy is not None:
        cat_policy = CatPolicy(cat_policy)
    if y_policy is not None:
        y_policy = YPolicy(y_policy)

    dataset = Dataset.from_dir(path, score)

    if dataset.X_num is None:
        assert num_policy is None
    elif num_policy is not None:
        dataset.data['X_num'] = transform_num(dataset.X_num, num_policy, seed)

    if dataset.X_cat is None:
        assert cat_policy is None
    elif cat_policy is not None:
        dataset.data['X_cat'] = transform_cat(dataset.X_cat, cat_policy)
        if cat_policy == CatPolicy.ONE_HOT:
            if dataset.X_num is None:
                dataset.data['X_num'] = dataset.data.pop('X_cat')
            else:
                dataset.data['X_num'] = {
                    k: np.concatenate([dataset.X_num[k], dataset.X_cat[k]], axis=1)
                    for k in dataset.X_num
                }
                dataset.data.pop('X_cat')

    if y_policy is not None:
        assert dataset.is_regression
        dataset.data['Y'], y_info = transform_y(dataset.Y, y_policy)
        dataset = replace(dataset, y_info=y_info)

    if cache_path is not None:
        util.dump_pickle((args, dataset), cache_path)
    return dataset


def are_valid_predictions(predictions: dict[str, np.ndarray]) -> bool:
    return all(np.isfinite(x).all() for x in predictions.values())
