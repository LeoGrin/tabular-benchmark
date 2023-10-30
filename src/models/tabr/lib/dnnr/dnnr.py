# fmt: off
# isort: off

import gc
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from torch import Tensor


@dataclass
class DNNR:
    n_neighbors: int = None
    n_derivative_neighbors: int = None
    n_jobs: Optional[int] = 1
    batch_size: Optional[int] = 1024

    def __post_init__(self):
        if isinstance(self.n_neighbors, int):
            self.n_neighbors = [self.n_neighbors]

    def fit(
        self,
        X_train: Union[Tensor, np.ndarray],
        y_train: Union[Tensor, np.ndarray],
        neighbors: Union[Tensor, np.ndarray],
    ):
        self.is_torch = isinstance(X_train, Tensor)
        if self.is_torch:
            self.device = X_train.device.type
            self.X_train = X_train
            self.y_train = y_train
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.X_train = torch.as_tensor(
                X_train, dtype=torch.float64, device=self.device
            )
            self.y_train = torch.as_tensor(
                y_train, dtype=torch.float64, device=self.device
            )
            neighbors = torch.as_tensor(neighbors, device="cpu")

        self.precompute_derivatives(neighbors)
        return self

    def precompute_derivatives(
        self,
        neighbors,
    ) -> None:
        idx_loader = torch.arange(self.X_train.shape[0], device="cpu").split(
            self.batch_size
        )
        derivatives = []
        for idx in idx_loader:
            x, y = self.X_train[idx], self.y_train[idx]
            nn_idx = neighbors[idx, :self.n_derivative_neighbors].to(self.device)
            derivatives.append(self._precompute_derivatives(x, y, nn_idx))
        self.derivatives = torch.cat(derivatives)

    def _precompute_derivatives(self, x: Tensor, y: Tensor, nn_idx: Tensor) -> Tensor:
        nn_x, nn_y = self.X_train[nn_idx], self.y_train[nn_idx]
        x_diff = (nn_x - x[:, None]).cpu().numpy()
        y_diff = (nn_y - y[:, None]).cpu().numpy()

        def linreg_fn(dx, dy):
            w = np.ones(dx.shape[0])
            solver = Ridge(
                alpha=1e-6,
                tol=1e-12,
                fit_intercept=False,
            ).fit(dx, dy, sample_weight=w)
            return solver.coef_

        derivatives = Parallel(n_jobs=self.n_jobs)(
            delayed(linreg_fn)(dx, dy) for dx, dy in zip(x_diff, y_diff)
        )
        return torch.as_tensor(np.stack(derivatives), device=self.device)

    def _predict(self, x: Tensor, nn_idx: Tensor) -> Tensor:
        nn_x, nn_y = (
            self.X_train[nn_idx],
            self.y_train[nn_idx],
        )
        derivatives = self.derivatives[nn_idx]

        x_diff = x[:, None] - nn_x
        first_order = (x_diff * derivatives).sum(2)
        predictions = (nn_y + first_order).mean(1)
        return predictions.cpu()

    def multi_predict(
        self, X_test: Union[Tensor, np.ndarray], neighbors: Union[Tensor, np.ndarray]
    ) -> Tensor:
        X_test = torch.as_tensor(X_test, dtype=torch.float64, device=self.device)
        neighbors = torch.as_tensor(neighbors, device="cpu")

        idx_loader = torch.arange(X_test.shape[0], device="cpu").split(
            self.batch_size
        )

        max_n = max(self.n_neighbors)
        predictions = {n: [] for n in self.n_neighbors}
        for idx in idx_loader:
            x = X_test[idx.to(self.device)]
            nn_idx = neighbors[idx, :max_n].to(self.device)

            for n in self.n_neighbors:
                predictions[n].append(self._predict(x, nn_idx[:, :n]))

        del X_test
        gc.collect()
        torch.cuda.empty_cache()

        for n in self.n_neighbors:
            predictions[n] = torch.cat(predictions[n])
            if not self.is_torch:
                predictions[n] = predictions[n].numpy()

        return predictions
