# fmt: off
# isort: off

from __future__ import annotations

import abc
import dataclasses
import tempfile
from typing import Any, Optional, TypeVar

import faiss
import numpy as np
import sklearn.base
import sklearn.neighbors

T = TypeVar('T')


class BaseIndex(sklearn.base.BaseEstimator, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, x: np.ndarray) -> None:
        """Builds the index.

        Args:
            x: with shape (n_samples, n_features). Used to build the
                index.
            **kwargs: Additional arguments passed to the index.
        """

    @abc.abstractmethod
    def query_knn(self, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the (indices, distances) of the k nearest neighbors of v.

        Args:
            v: with shape (n_samples, n_features)
            k: number of neighbors to return

        Returns:
            A tuple of (indices, distances) of the k nearest neighbors of v.
        """


@dataclasses.dataclass
class L2Index(BaseIndex):
    def fit(self, X_train: np.ndarray) -> None:
        X_train = np.ascontiguousarray(X_train)
        self.index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),
            0,
            faiss.IndexFlatL2(X_train.shape[1]),
        )
        self.index.add(X_train.astype('float32'))

    def query_knn(self, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "index"):
            raise ValueError("Index not fitted.")
        distances, indices = self.index.search(v[None].astype('float32'), k)
        return indices[0], distances[0][0]


@dataclasses.dataclass
class KDTreeIndex(BaseIndex):
    metric: str = "euclidean"
    leaf_size: int = 40
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def fit(self, x: np.ndarray) -> None:
        self.index = sklearn.neighbors.KDTree(
            x, metric=self.metric, leaf_size=self.leaf_size, **self.kwargs
        )

    def query_knn(self, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "index"):
            raise ValueError("Index not fitted.")
        distances, indices = self.index.query([v], k=k)
        return indices[0], distances[0][0]


@dataclasses.dataclass
class AnnoyIndex(BaseIndex):
    metric: str = "euclidean"
    n_trees: int = 50
    n_features: Optional[int] = None

    def fit(self, x: np.ndarray) -> None:
        import annoy

        self.n_features = x.shape[1]
        self.index = annoy.AnnoyIndex(self.n_features, self.metric)

        for i, v in zip(range(len(x)), x):
            self.index.add_item(i, v)
        self.index.build(self.n_trees)

    def query_knn(self, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "index"):
            raise ValueError("Index not fitted.")

        indices, distances = self.index.get_nns_by_vector(v, k, include_distances=True)
        return indices, distances

    def __getstate__(self):
        state = self.__dict__.copy()

        with tempfile.TemporaryDirectory() as dir:
            fname = dir + "/index.ann"
            self.index.save(fname)
            with open(fname, "rb") as f:
                state["index"] = f.read()
        return state

    def __setstate__(self, state):
        import annoy

        index_bytes = state.pop("index")
        self.index = annoy.AnnoyIndex(state['n_features'], state['metric'])

        with tempfile.NamedTemporaryFile() as f:
            f.write(index_bytes)
            f.flush()
            self.index.load(f.name)

        self.__dict__.update(state)


def get_index_class(index: type[BaseIndex] | str) -> type[BaseIndex]:
    """Returns the corresponding index class based on the passed string.

    Args:
        index: either a string of the index name or a class
    """
    if isinstance(index, type) and issubclass(index, BaseIndex):
        return index

    if index == "l2":
        return L2Index
    elif index == "annoy":
        return AnnoyIndex
    elif index == "kd_tree":
        return KDTreeIndex
    else:
        raise ValueError(f"Index {index} not supported")
