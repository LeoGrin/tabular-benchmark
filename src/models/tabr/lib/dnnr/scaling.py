"""This module contains the input scaling."""

# fmt: off
# isort: off

import abc
import dataclasses
import random as random_mod
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import scipy.optimize
import scipy.spatial.distance
import sklearn.base
import tqdm.auto as tqdm
from sklearn import model_selection

from lib.dnnr import nn_index


class InputScaling(sklearn.base.BaseEstimator, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns the scaling vector of the input.

        Args:
            X_train: The training data.
            y_train: The training targets.
            X_test: The test data.
            y_test: The test targets.

        Returns:
            The scaling vector.
        """

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    @abc.abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the input.

        Args:
            X: The input.

        Returns:
            The transformed input.
        """

class _Optimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, gradients: List[np.ndarray]) -> None:
        """Updates the parameters.

        Args:
            gradients: The gradients of the parameters.
        """


@dataclasses.dataclass
class SGD(_Optimizer):
    """Stochastic gradient descent optimizer.

    Args:
        parameters: The parameters to optimize.
        lr: The learning rate.
    """

    parameters: List[np.ndarray]
    lr: float = 0.01

    def step(self, gradients: List[np.ndarray]) -> None:
        for param, grad in zip(self.parameters, gradients):
            param -= self.lr * grad


@dataclasses.dataclass
class RMSPROP:
    """The RMSPROP optimizer.

    Args:
        parameters: The parameters to optimize.
        lr: The learning rate.
        γ: The decay rate.
        eps: The epsilon to avoid division by zero.
    """

    parameters: List[np.ndarray]
    lr: float = 1e-4
    γ: float = 0.99
    eps: float = 1e-08

    def __post_init__(self):
        self.v = [np.zeros_like(param) for param in self.parameters]

    def step(self, gradients: List[np.ndarray]) -> None:
        for param, grad, v in zip(self.parameters, gradients, self.v):
            # inplace update
            v[:] = self.γ * v + (1 - self.γ) * grad**2
            update = self.lr * grad / (np.sqrt(v) + self.eps)
            param -= update


@dataclasses.dataclass
class LearnedScaling(InputScaling):
    """This class handles the scaling of the input.

    Args:
        n_epochs: The number of epochs to train the scaling.
        optimizer: The optimizer to use (either `SGD` or `RMSPROP`).
        optimizer_params: The parameters of the optimizer.
        epsilon: The epsilon for gradient computation.
        random: The `random.Random` instance for this class.
        show_progress: Whether to show a progress bar.
        fail_on_nan: Whether to fail on NaN values.
    """

    n_epochs: int = 1
    optimizer: Union[str, Type[_Optimizer]] = SGD
    optimizer_params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    shuffle: bool = True
    epsilon: float = 1e-6
    random: random_mod.Random = dataclasses.field(
        default_factory=lambda: random_mod.Random(random_mod.randint(0, 2**32 - 1))
    )
    show_progress: bool = False
    fail_on_nan: bool = False
    index: Union[str, Type[nn_index.BaseIndex]] = 'l2'
    index_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.scaling_: Optional[np.ndarray] = None
        self.scaling_history: list = []
        self.scores_history: list = []
        self.costs_history: list = []
        self.index_cls = nn_index.get_index_class(self.index)
        self._fitted: bool = False

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self.scaling_ is None:
            raise RuntimeError("Not fitted")
        return X * self.scaling_

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_neighbors: Optional[int] = None,
        val_size: Optional[int] = None,
    ) -> np.ndarray:
        """Fits the scaling vector.

        Args:
            X_train: The training data.
            y_train: The training targets.
            X_val: The validation data.
            y_val: The validation targets.
            val_size: The size of the validation set.

        If the validation set is not provided, the training set is split into
        a validation set using the `val_size` parameter.

        Returns:
            The scaling vector.
        """
        # contiguousness is important for faiss
        X_train = np.ascontiguousarray(X_train)
        if X_val is not None:
            X_val = np.ascontiguousarray(X_val)

        n_features = X_train.shape[1]
        batch_size = 8 * n_features if n_neighbors is None else n_neighbors
        scaling = np.ones((1, n_features))

        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be either given or not.")

        if X_val is None and y_val is None:
            split_size = val_size if val_size is not None else int(0.1 * len(X_train))
            if split_size < 10:
                warnings.warn(
                    "Validation split for scaling is small! Scaling is skipped!"
                    f" Got {split_size} samples."
                )
                # do not scale
                self.scaling_ = scaling
                self._fitted = True
                return scaling
            X_train, X_val, y_train, y_val = model_selection.train_test_split(
                X_train,
                y_train,
                test_size=split_size,
                random_state=self.random.randint(0, 2**32 - 1),
            )

        assert X_val is not None
        assert y_val is not None

        def handle_possible_nans(grad: np.ndarray) -> bool:
            if not np.isfinite(grad).all():
                if self.fail_on_nan:
                    raise RuntimeError("Gradient contains NaN or Inf")

                warnings.warn("Found inf/nans in gradient. " "Scaling is returned now.")
                return True
            else:
                return False

        def get_optimizer() -> _Optimizer:
            if isinstance(self.optimizer, str):
                optimizer_cls = {
                    'sgd': SGD,
                    'rmsprop': RMSPROP,
                }[self.optimizer.lower()]
            else:
                optimizer_cls = self.optimizer

            kwargs = self.optimizer_params.copy()
            kwargs['parameters'] = scaling
            return optimizer_cls(**kwargs)

        if self._fitted:
            raise RuntimeError("Already fitted scaling vector")

        self._fitted = True

        optimizer = get_optimizer()

        for epoch in range(self.n_epochs):
            index = self.index_cls(**self.index_kwargs)
            index.fit(scaling * X_train)

            train_index = list(range(len(X_train)))
            if self.shuffle:
                self.random.shuffle(train_index)
            for idx in tqdm.tqdm(train_index, desc=f'epoch {epoch}'):
                v = X_train[idx]
                y = y_train[idx]
                indices, _ = index.query_knn(v * scaling[0], batch_size)
                # skip `v` itself
                indices = indices[1:]
                nn_x = X_train[indices]
                nn_y = y_train[indices]

                cost, grad = self._get_gradient(scaling, nn_x, nn_y, v, y)

                if handle_possible_nans(grad):
                    return self.scaling_history

                self.costs_history.append(cost)
                optimizer.step([grad])
            self.scaling_history.append(scaling.copy())

        return self.scaling_history

    def _get_gradient(
        self,
        scaling: np.ndarray,
        nn_x: np.ndarray,
        nn_y: np.ndarray,
        v: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the loss and the gradient.

        Args:
            scaling: The scaling vector.
            nn_x: The nearest neighbors of the current sample.
            nn_y: The targets of the nearest neighbors.
            v: The current sample.
            y: The target of the current sample.
        """
        q = nn_y - y
        delta = nn_x - v
        try:
            pinv = np.linalg.pinv(delta.T @ delta)
            nn_y_hat = pinv @ (delta.T @ q)
        except RuntimeError:
            raise RuntimeError(
                "Failed to compute psydo inverse!" f" The scaling vector was: {scaling}"
            )

        y_pred = y + delta @ nn_y_hat.T
        scaled_nn_x = nn_x * scaling
        scaled_v = v * scaling

        h_norm_in = scaled_nn_x - scaled_v
        h = np.clip(np.linalg.norm(h_norm_in, axis=1), self.epsilon, None)

        q = np.abs(nn_y - y_pred)

        vq = q - np.mean(q)
        vh = h - np.mean(h)

        if np.allclose(vq, 0) or np.allclose(vh, 0):
            # Either vq are all equal (this can be the case if the target value
            # is the same for all samples). Or vh are all equal to 0, which can
            # happen if the nearest neighbors are all the same. In any case, we
            # can't compute the cosine similarity in this case and therefore
            # return 0.
            return np.array([0.0]), np.zeros(scaling.shape[1])

        cossim = self._cossim(vq, vh)
        cost = -cossim
        # Backward path

        dcossim = -np.ones(1)  # ensure to account for - cossim
        _, dvh = self._cossim_backward(dcossim, cossim, vq, vh)

        # Derive: vh = h - np.mean(h)
        # d vh_j / d h_i =  - 1 / len(h)  if i != j
        # d vh_j / d h_i =  1 - 1 / len(h) if i == j
        #  -> I - 1/len(h)
        len_h = np.prod(h.shape)
        dim = dvh.shape[0]
        mean_len_matrix = np.full(dim, dim, 1 / len_h)
        mean_jac = np.eye(dim) - mean_len_matrix
        # dh = (1. - 1 / mean_len) * dvh
        dh = mean_jac @ dvh

        dh_norm_in = self._l2_norm_backward(dh, h, h_norm_in)

        # Derive: h_norm_in = scaled_nn_x - scaled_v
        dscaled_nn_x = dh_norm_in
        dscaled_v = -dh_norm_in

        # Derive: scaled_nn_x = nn_x * fsv
        dfsv_nn_x = nn_x * dscaled_nn_x
        # Derive: scaled_v = v * fsv
        dfsv_v = v * dscaled_v

        # Accumulate gradients
        dfsv = dfsv_nn_x + dfsv_v
        return cost, dfsv.sum(axis=0)

    @staticmethod
    def _l2_norm_backward(
        grad: np.ndarray, l2_norm: np.ndarray, a: np.ndarray
    ) -> np.ndarray:
        """Backward pass for the l2 norm.

        Args:
            grad: The backpropaged gradient.
            l2_norm: The l2 norm of the input.
            a: The input to the l2 norm.
        """
        # From: https://en.wikipedia.org/wiki/Norm_(mathematics)
        # d(||a||_2) / da = a / ||a||_2
        da = a / l2_norm[:, np.newaxis]
        return da * grad[:, np.newaxis]

    @staticmethod
    def _cossim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Computes the cosine similarity between two vectors."""
        return 1 - scipy.spatial.distance.cosine(a, b)

    @staticmethod
    def _cossim_backward(
        grad: np.ndarray,
        cossim: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Backward pass for the cosine similarity.

        Args:
            grad: The backpropaged gradient.
            cossim: The cosine similarity of the input.
            a: The first input to the cosine similarity.
            b: The second input to the cosine similarity.
            eps: The epsilon to avoid numerical issues.

        Returns:
            A tuple of the gradient of the first input and the gradient of the
            second input.
        """
        # From: https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosine-similarity  # noqa
        #
        # d/da_i cossim(a, b) = b_i / (|a| |b|) - cossim(a, b) * a_i / |a|^2
        # analogously for b
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        dcos_da = (b / (na * nb + eps)) - (cossim * a / (na**2))
        dcos_db = (a / (na * nb + eps)) - (cossim * b / (nb**2))
        return dcos_da * grad, dcos_db * grad
