# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import delu
import numpy as np
import torch
from category_encoders import LeaveOneOutEncoder

import lib
from lib import KWArgs


@dataclass(frozen=True)
class Config:
    seed: int
    data: KWArgs  # lib.data.build_dataset
    n_neighbors: int


@torch.inference_mode()
def main(
    config: lib.JSONDict, output: Union[str, Path], *, force: bool = False
) -> Optional[lib.JSONDict]:
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    delu.random.seed(C.seed)
    device = lib.get_device()

    dataset = lib.build_dataset(**C.data)

    embeddings = {'train': [], 'val': [], 'test': []}
    if dataset.X_num is not None:
        for part in dataset.parts():
            embeddings[part].append(dataset.X_num[part])
    if dataset.X_bin is not None:
        for part in dataset.parts():
            embeddings[part].append(dataset.X_bin[part])
    if dataset.X_cat is not None:
        assert (
            not dataset.is_multiclass
        ), 'For multiclass problems, categorical features are not supported'
        if dataset.is_regression:
            assert C.data['y_policy'] == 'standard'
        cat_encoder = LeaveOneOutEncoder(
            cols=list(range(dataset.n_cat_features)),
            sigma=0.1,
            random_state=C.seed,
            return_df=False,
        )
        cat_encoder.fit(dataset.X_cat['train'], dataset.Y['train'].astype('float32'))
        for part in dataset.parts():
            embeddings[part].append(cat_encoder.transform(dataset.X_cat[part]))
    embeddings = {
        k: np.concatenate(v, dtype='float32', axis=-1) for k, v in embeddings.items()
    }
    embeddings = {
        k: torch.as_tensor(v, dtype=torch.float32, device=device)
        for k, v in embeddings.items()
    }

    lib.save_knn(*lib.compute_knn(embeddings, C.n_neighbors, verbose=True), output)
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)
