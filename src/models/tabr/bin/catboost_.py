# CatBoost

# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import delu
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from loguru import logger

import lib
from lib import KWArgs


@dataclass(frozen=True)
class Config:
    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]  # lib.data.build_dataset
    model: KWArgs  # catboost.{CatBoostClassifier,CatBoostRegressor}
    fit: KWArgs  # catboost.{CatBoostClassifier,CatBoostRegressor}.fit

    def __post_init__(self):
        assert 'random_seed' not in self.model
        assert (
            self.model.get('l2_leaf_reg', 3.0) > 0
        )  # CatBoost fails on multiclass problems with l2_leaf_reg=0
        assert (
            'task_type' in self.model
        ), 'task_type significantly affects performance, so must be set explicitly'
        if self.model['task_type'] == 'GPU':
            assert os.environ.get('CUDA_VISIBLE_DEVICES')
        else:
            assert not os.environ.get('CUDA_VISIBLE_DEVICES')


def main(
    config: lib.JSONDict, output: Union[str, Path], *, force: bool = False
) -> Optional[lib.JSONDict]:
    # >>> start
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    delu.random.seed(C.seed)

    # >>> data
    dataset = C.data if isinstance(C.data, lib.Dataset) else lib.build_dataset(**C.data)
    dataset = dataset.merge_num_bin()
    if dataset.X_num is None:
        assert dataset.X_cat is not None
        X = {k: pd.DataFrame(v) for k, v in dataset.X_cat.items()}
    elif dataset.X_cat is None:
        assert dataset.X_num is not None
        X = {k: pd.DataFrame(v) for k, v in dataset.X_num.items()}
    else:
        X = {
            part: pd.concat(
                [
                    pd.DataFrame(
                        dataset.X_num[part], columns=range(dataset.n_num_features)
                    ),
                    pd.DataFrame(
                        dataset.X_cat[part],
                        columns=range(dataset.n_num_features, dataset.n_features),
                    ),
                ],
                axis=1,
            )
            for part in dataset.parts()
        }

    # >>> model
    model_kwargs = C.model | {
        'random_seed': C.seed,
        'train_dir': output / 'catboost_info',
    }
    if dataset.X_cat is not None:
        model_kwargs['cat_features'] = list(
            range(dataset.n_num_features, dataset.n_features)
        )
    if C.model['task_type'] == 'GPU':
        C.model['devices'] = '0'
    if dataset.is_regression:
        model = CatBoostRegressor(**model_kwargs)
        predict = model.predict
    else:
        model = CatBoostClassifier(**model_kwargs, eval_metric='Accuracy')
        predict = (
            model.predict_proba
            if dataset.is_multiclass
            else lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]
        )
    report['prediction_type'] = None if dataset.is_regression else 'probs'

    # >>> training
    logger.info('training...')
    with delu.Timer() as timer:
        model.fit(
            X['train'],
            dataset.Y['train'],
            eval_set=(X['val'], dataset.Y['val']),
            **C.fit
        )
    shutil.rmtree(model_kwargs['train_dir'])
    report['time'] = str(timer)
    report['best_iteration'] = model.get_best_iteration()

    # >>> finish
    model.save_model(output / 'model.cbm')
    np.save(output / "feature_importances.npy", model.get_feature_importance())
    predictions = {k: predict(v) for k, v in X.items()}
    report['metrics'] = dataset.calculate_metrics(
        predictions, report['prediction_type']  # type: ignore[code]
    )
    lib.dump_predictions(predictions, output)
    lib.dump_summary(lib.summarize(report), output)
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)
