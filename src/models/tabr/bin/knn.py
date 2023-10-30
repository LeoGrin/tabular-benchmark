# k Nearest Neighbors

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
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import delu
import numpy as np
from loguru import logger
from tqdm import tqdm

import lib


@dataclass
class Config:
    @dataclass
    class Data:
        path: Path
        score: Optional[str] = None

    data: Data
    neighbors: Path
    k_range: list[int]


def main(
    config: lib.JSONDict,
    output: Union[str, Path],
    *,
    force: bool = False,
) -> Optional[lib.JSONDict]:
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    delu.random.seed(0)  # just in case :)

    dataset = lib.Dataset.from_dir(lib.get_path(C.data.path), C.data.score)
    report['prediction_type'] = None if dataset.is_regression else 'probs'
    neighbor_Y = {
        part: dataset.Y['train'][
            np.load(lib.get_path(C.neighbors) / f"neighbors_{part}.npy")
        ]
        for part in dataset.parts()
    }

    best_score = -math.inf
    best_predictions: Optional[dict[str, np.ndarray]] = None
    metrics = []
    for n_neighbors in tqdm(range(*C.k_range), desc='Tuning k'):
        if dataset.is_regression:
            predictions = {
                part: neighbor_Y[part][:, :n_neighbors].mean(1)
                for part in dataset.parts()
            }
        else:
            predictions = {
                part: np.column_stack(
                    [
                        (neighbor_Y[part][:, :n_neighbors] == class_id).sum(1)
                        / n_neighbors
                        for class_id in (
                            range(dataset.n_classes()) if dataset.is_multiclass else [1]
                        )
                    ]
                )
                for part in dataset.parts()
            }
            if dataset.is_binclass:
                predictions = {k: v.squeeze(1) for k, v in predictions.items()}

        metrics.append(
            dataset.calculate_metrics(predictions, report['prediction_type'])
        )
        score = metrics[-1]['val']['score']
        if score > best_score:
            best_score = score
            best_predictions = predictions
            report['n_neighbors'] = n_neighbors
            report['metrics'] = deepcopy(metrics[-1])

    assert best_predictions is not None
    logger.info(f'Best k = {report["n_neighbors"]}')
    lib.dump_json(metrics, output / 'all_metrics.json')
    lib.dump_predictions(best_predictions, output)
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)
