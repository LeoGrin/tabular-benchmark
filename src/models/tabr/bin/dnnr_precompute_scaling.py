# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import json
from argparse import ArgumentParser
from collections import defaultdict
from time import time

import delu
import numpy as np
import tomli
from category_encoders import LeaveOneOutEncoder
from loguru import logger
from tqdm import tqdm

import lib
from lib.dnnr import PrecomputeLearnedScaling

lib.configure_libraries()

# process config
parser = ArgumentParser(
    description="""Let config be the config under the config_path path.
    Then config["dnnr"]["scalings_dir"] is a path to dir, where computed scalings will be stored.
    The bin/dnnr.py will pull up the scalings from there and use aforementioned config for evaluation.
    """,
)
parser.add_argument("config_path", metavar="config_path", type=str)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument(
    "--n_neighbors_list",
    type=int,
    nargs="*",
    help="""If not provided the some default n_neighbors_list will be formed.
    If provided and no arguments are given, scaling for only the default number of neighbors (from original paper) will be computed.
    If provided and arguments are given, scalings for only given number of neighbors will be computed.
    """,
)
parser.add_argument(
    "--dont_add_default_n",
    action="store_true",
    help="""False: to previously formed list of n_neighbors the default value (from original paper) will be added.
    True: won't be added.
    """,
)
args = parser.parse_args()

assert args.n_neighbors_list or not args.dont_add_default_n

with open(lib.PROJECT_DIR / args.config_path, "rb") as file:
    config = tomli.load(file)

scalings_dir = lib.EXP_DIR / config["dnnr"]["scalings_dir"]

# prepare data
delu.random.seed(0)
for k, v in config["data"].items():
    if v == "__null__":
        config["data"][k] = None

dataset = lib.build_dataset(**config["data"])

parts = {"train", "val", "test"}
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

# prepare n_neighbors_list
n_neighbors_list = args.n_neighbors_list
if n_neighbors_list is None:
    n_neighbors_list = [3, 96] + [2**i for i in range(1, 8)]
if not args.dont_add_default_n:
    n_neighbors_list = [None] + n_neighbors_list

for i, n_neighbors in enumerate(tqdm(n_neighbors_list, desc="n neighbors loop")):
    report = defaultdict(dict)

    start_time = time()
    scalings = PrecomputeLearnedScaling(n_epochs=args.n_epochs).fit(
        X["train"], Y["train"], X["val"], Y["val"], n_neighbors=n_neighbors
    )
    report["n_neighbors"] = n_neighbors
    report["time"] = time() - start_time
    report["n_scalings_all"] = len(scalings)

    # save scalings
    if n_neighbors is None:
        n_neighbors = "default"

    k_dir = scalings_dir / str(n_neighbors)
    k_dir.mkdir(exist_ok=True, parents=True)

    n_scalings_skiped = 0
    for epoch, scaling in enumerate(scalings):
        if (scaling == np.ones(X["train"].shape[1])).all():
            n_scalings_skiped += 1
            continue

        with open(k_dir / f"{epoch + 1}.npy", "wb") as file:
            np.save(file, scaling.reshape(-1))

    report["n_scalings_not_ones"] = len(scalings) - n_scalings_skiped

    with open(k_dir / "report.json", "w") as file:
        json.dump(report, file, indent=4)

    logger.info(f"{i + 1} / {len(n_neighbors_list)} ")
    logger.info(f"report_path: {k_dir / 'report.json'}")
