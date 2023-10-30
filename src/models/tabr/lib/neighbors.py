from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from . import util


# NOTE:
# - this is a naive function implemented via torch; in practice, use faiss.
# - the function always returns CPU tensors regardless of the original device.
def compute_knn(
    embeddings: dict[
        str, Tensor
    ],  # {'train': train_embeddings, 'val': ..., 'test': ...}
    n_neighbors: int,
    batch_size: int = 1024,
    verbose: bool = False,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:  # (neighbors, distances)
    logger.info('Computing k nearest neighbors')

    assert 'train' in embeddings
    device = next(iter(embeddings.values())).device
    neighbors = {}
    distances = {}
    while True:
        try:
            for part in embeddings:
                distances[part] = []
                neighbors[part] = []
                batches = torch.arange(len(embeddings[part]), device=device).split(
                    batch_size
                )
                if verbose:
                    batches = tqdm(batches, desc=part)
                for batch_idx in batches:
                    batch_distances = torch.cdist(
                        embeddings[part][batch_idx][None], embeddings['train'][None]
                    ).squeeze(0)
                    if part == 'train':
                        batch_distances[
                            torch.arange(len(batch_idx)), batch_idx
                        ] = torch.inf
                    topk = torch.topk(
                        batch_distances, n_neighbors, dim=1, largest=False
                    )
                    distances[part].append(topk.values.cpu())
                    neighbors[part].append(topk.indices.cpu())
        except RuntimeError as err:
            if util.is_oom_exception(err):
                batch_size //= 2
                logger.warning(f'compute_knn: batch_size = {batch_size}')
            else:
                raise
        else:
            break

    for data in [neighbors, distances]:
        for key in list(data):
            data[key] = torch.cat(data[key])
    return neighbors, distances


def save_knn(
    neighbors: dict[str, Tensor], distances: dict[str, Tensor], path: Path
) -> None:
    assert path.is_dir()
    for part in neighbors:
        np.save(path / f'distances_{part}.npy', distances[part].cpu().numpy())
        np.save(
            path / f'neighbors_{part}.npy',
            neighbors[part].cpu().numpy().astype('int32'),  # to save space
        )
