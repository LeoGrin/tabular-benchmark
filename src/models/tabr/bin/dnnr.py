# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import gc
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, Optional, Union

import delu
import faiss
import numpy as np
import torch
from category_encoders import LeaveOneOutEncoder
from loguru import logger
from tqdm import tqdm

import lib
from lib import KWArgs
from lib.dnnr import DNNR


@dataclass
class Config:
    @dataclass
    class ModelConfig:
        n_neighbors: Union[list[int], int]
        n_derivative_neighbors: Union[list[int], int]
        scalings_dir: str
        n_jobs: Optional[int] = 8
        batch_size: Optional[int] = 1024

    data: Union[lib.Dataset[np.ndarray], KWArgs]
    dnnr: ModelConfig
    add_ones_scaling: Optional[bool] = True
    seed: Optional[int] = 0


def get_scalings_from_dir(scalings_dir: Path) -> Dict:
    scalings = {}  # n_epoch_name -> scalings
    for n_epoch_name in os.listdir(scalings_dir):
        if n_epoch_name == "report.json":
            continue
        n_epoch = n_epoch_name.rstrip(".npy")
        with open(scalings_dir / n_epoch_name, "rb") as file:
            scalings[n_epoch] = np.load(file)
    return scalings


def compute_knn(
    X: Dict[str, np.ndarray],
    n_neighbors: int,
    batch_size=512,
) -> Dict[str, np.ndarray]:
    X = {k: v.astype("float32") for k, v in X.items()}
    neighbors = defaultdict(list)

    index = faiss.index_cpu_to_gpu(
        faiss.StandardGpuResources(),
        0,
        faiss.IndexFlatL2(X["train"].shape[1]),
    )
    index.add(X["train"])

    for part in X:
        is_train = part == "train"

        idx_loader = np.array_split(np.arange(X[part].shape[0]), batch_size)
        for idx in idx_loader:
            _, indices = index.search(
                X[part][idx],
                n_neighbors + (1 if is_train else 0),
            )
            if is_train:
                neighbors[part].append(indices[:, 1:])
            else:
                neighbors[part].append(indices)

    neighbors = {k: np.concatenate(v) for k, v in neighbors.items()}
    return neighbors


def main(
    config: Union[Config, lib.JSONDict],
    output: Union[str, Path],
    *,
    force: bool = False,
) -> Optional[lib.JSONDict]:
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    C = lib.make_config(Config, config)

    report = lib.create_report(C)
    report["config"] = config
    delu.random.seed(C.seed)

    dataset = lib.build_dataset(**C.data)

    # prepare data
    parts = {'train', 'val', 'test'}
    embeddings = {p: [] for p in parts}
    if dataset.X_num is not None:
        for part in parts:
            embeddings[part].append(dataset.X_num[part])
    if dataset.X_bin is not None:
        for part in parts:
            embeddings[part].append(dataset.X_bin[part])
    if dataset.X_cat is not None:
        cat_encoder = LeaveOneOutEncoder(
            cols=list(range(dataset.n_cat_features)),
            sigma=0.1,
            random_state=0,
            return_df=False,
        )
        cat_encoder.fit(dataset.X_cat["train"], dataset.Y["train"].astype("float64"))
        for part in parts:
            embeddings[part].append(cat_encoder.transform(dataset.X_cat[part]))

    X = {k: np.concatenate(v, dtype="float64", axis=-1) for k, v in embeddings.items()}
    Y = {part: dataset.Y[part].astype("float64") for part in parts}

    # prepare scalings
    scalings = defaultdict(dict)  # n_neighbors_name -> n_epoch -> scaling
    if C.dnnr.scalings_dir == "<no_scaling>":
        scalings["None"]["None"] = np.ones(X["train"].shape[1])
    else:
        # assumed that only the following dir types could be submitted for entry
        #     * scaling (.npy)
        #     * dir of sclaings
        #     * dir of dir of scalings
        #         those dirs are understood as:
        #             scaling_graph_type -> n_sclaing_neighbors -> n_epoch.npy

        scalings_dir = lib.EXP_DIR / C.dnnr.scalings_dir
        assert scalings_dir.exists(), scalings_dir

        if scalings_dir.is_file():
            with open(scalings_dir, 'rb') as file:
                scalings["None"]["None"] = np.load(file)
        else:
            if C.add_ones_scaling:
                scalings["no_scaling"][0] = np.ones(X["train"].shape[1])

            subdir_name = os.listdir(scalings_dir)[0]
            if (scalings_dir / subdir_name).is_file():
                scalings["None"] = get_scalings_from_dir(scalings_dir)
            else:
                for n_scaling_neighbors_name in os.listdir(scalings_dir):
                    n_scaling_neighbors_dir = scalings_dir / n_scaling_neighbors_name

                    scaling_report_path = n_scaling_neighbors_dir / "report.json"
                    if scaling_report_path.exists():
                        with open(scaling_report_path) as file:
                            scaling_report = json.load(file)
                            if scaling_report["n_scalings_not_ones"] == 0:
                                continue

                    scalings[n_scaling_neighbors_name] = get_scalings_from_dir(
                        n_scaling_neighbors_dir
                    )

    # prepare n_neighbors_lists
    if isinstance(C.dnnr.n_neighbors, int):
        n_neighbors_list = [C.dnnr.n_neighbors]
    else:
        n_neighbors_list = list(range(*C.dnnr.n_neighbors))

    if isinstance(C.dnnr.n_derivative_neighbors, int):
        n_derivative_neighbors_list = [C.dnnr.n_derivative_neighbors]
    else:
        n_derivative_neighbors_list = list(
            map(int, np.round(np.linspace(*C.dnnr.n_derivative_neighbors)).astype(int))
        )

    parts = ["val", "test"]
    score_history = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )  # n_derivative_neighbors -> n_scaling_neighbors_name -> n_epoch -> n_neighbors -> part -> score

    max_n = max(
        max(n_derivative_neighbors_list),
        max(n_neighbors_list),
    )

    for i, (n_scaling_neighbors_name, n_epoch_scaling_dict) in enumerate(
        scalings.items()
    ):
        for n_epoch, scaling in tqdm(
            n_epoch_scaling_dict.items(), desc=f"{i+1}/{len(scalings)}: "
        ):
            X_scaled = {
                k: np.ascontiguousarray(scaling[None] * v).astype("float64")
                for k, v in X.items()
            }
            neighbors = compute_knn(X_scaled, max_n, batch_size=C.dnnr.batch_size)

            for i, n_derivative_neighbors in enumerate(n_derivative_neighbors_list):
                start_time = time()

                # model init
                dnnr_config = {
                    "n_neighbors": n_neighbors_list,
                    "n_derivative_neighbors": n_derivative_neighbors,
                    "n_jobs": C.dnnr.n_jobs,
                    "batch_size": C.dnnr.batch_size,
                }

                # train
                model = DNNR(**dnnr_config).fit(
                    X_scaled["train"],
                    Y["train"],
                    neighbors["train"],
                )

                # eval
                preds = {
                    part: model.multi_predict(X_scaled[part], neighbors[part])
                    for part in parts
                }  # part -> n_neighbors -> pred
                del model
                gc.collect()
                torch.cuda.empty_cache()

                scores = defaultdict(dict)  # n_neighbors -> part -> score
                for part in parts:
                    for n_neighbors, pred in preds[part].items():
                        rmse = lib.calculate_metrics(
                            y_true=dataset.Y[part],
                            y_pred=pred,
                            task_type=dataset.task_type,
                            prediction_type=None,
                            y_std=dataset.y_info.std,
                        )['rmse']
                        scores[n_neighbors][part] = -rmse

                # store results
                for n_neighbors in scores:
                    if (
                        "metrics" not in report
                        or report["metrics"]["val"] < scores[n_neighbors]["val"]
                    ):
                        report["best_n_derivative_neighbors"] = n_derivative_neighbors
                        report[
                            "best_n_scaling_neighbors_name"
                        ] = n_scaling_neighbors_name
                        report["best_n_epoch"] = n_epoch
                        report["best_n_neighbors"] = n_neighbors
                        report["time"] = time() - start_time
                        report["metrics"] = {
                            part: scores[n_neighbors][part] for part in parts
                        }

                score_history[n_derivative_neighbors][n_scaling_neighbors_name][
                    n_epoch
                ] = scores
                with open(output / "score_history.json", "w") as file:
                    json.dump(score_history, file, indent=4)
                    file.flush()

    logger.info(f"metrics: {report['metrics']}")
    report["metrics"] = {
        part: {"score": val} for part, val in report["metrics"].items()
    }
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)
