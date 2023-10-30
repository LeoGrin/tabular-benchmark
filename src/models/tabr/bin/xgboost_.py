# XGBoost

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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import delu
import numpy as np
from loguru import logger
from xgboost import XGBClassifier, XGBRegressor

import lib
from lib import KWArgs


# DO NOT modify the config neither in __post_init__ nor during execution
@dataclass(frozen=True)
class Config:
    @dataclass(frozen=True)
    class Neighbors:
        path: Path
        k: int

    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]  # lib.data.build_dataset
    neighbors: Optional[Neighbors]
    model: KWArgs  # xgboost.{XGBClassifier,XGBRegressor}
    fit: KWArgs  # xgboost.{XGBClassifier,XGBRegressor}.fit

    def __post_init__(self):
        assert 'random_state' not in self.model
        if isinstance(self.data, dict):
            assert self.data['cat_policy'] in [None, 'one-hot']
        assert (
            'early_stopping_rounds' in self.fit
        ), 'XGBoost does not automatically use the best model, so early stopping must be used'
        use_gpu = self.model.get('tree_method') == 'gpu_hist'
        if use_gpu:
            assert os.environ.get('CUDA_VISIBLE_DEVICES')
        else:
            assert not os.environ.get('CUDA_VISIBLE_DEVICES')


def _patch_config(c):
    c.setdefault('neighbors', None)


def main(
    config: lib.JSONDict, output: Union[str, Path], *, force: bool = False
) -> Optional[lib.JSONDict]:
    # >>> start
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    # all modifications to `config` must be done BEFORE creating the report
    _patch_config(config)
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    delu.random.seed(C.seed)

    # >>> data
    dataset = C.data if isinstance(C.data, lib.Dataset) else lib.build_dataset(**C.data)
    dataset = dataset.merge_num_bin()
    assert set(dataset.data.keys()) == {'X_num', 'Y'}, set(dataset.data.keys())
    assert dataset.X_num is not None  # for type checker
    if C.neighbors is not None:
        logger.info('Adding neighbor features')
        new_X_num = {}
        for part in dataset.parts():
            neighbors = np.load(
                lib.get_path(C.neighbors.path) / f"neighbors_{part}.npy"
            )[:, : C.neighbors.k]
            neighbor_Y = dataset.Y['train'][neighbors]
            if dataset.is_classification:
                dtype = dataset.X_num['train'].dtype
                class_ids = np.arange(dataset.n_classes())
                assert (class_ids == np.unique(dataset.Y['train'])).all()
                # neighbor class distribution
                neighbor_label_features = (
                    neighbor_Y[:, :, None] == class_ids[None, None]
                ).astype(dtype).sum(1) / C.neighbors.k
                if dataset.is_binclass:
                    neighbor_label_features = neighbor_label_features[:, 1:]
            elif dataset.is_regression:
                assert config['data']['y_policy'] == 'standard'
                neighbor_label_features = neighbor_Y.mean(1, keepdims=True)
            else:
                raise ValueError('Unknown task type')
            new_X_num[part] = np.concatenate(
                [
                    dataset.X_num[part],
                    dataset.X_num['train'][neighbors].mean(1),
                    neighbor_label_features,
                ],
                axis=1,
            )
        dataset.data['X_num'] = new_X_num

    # >>> model
    # NOTE: we use Scikit-Learn API AND we require "early_stopping_rounds" to be
    # explicitely provided. In this case, the `.predict[_proba]` methods
    # automatically use the best model.
    model_extra_kwargs = {'random_state': C.seed}
    if dataset.is_regression:
        model = XGBRegressor(**C.model, **model_extra_kwargs)
        predict = model.predict
        fit_extra_kwargs = {}
    else:
        model = XGBClassifier(
            **C.model, **model_extra_kwargs, disable_default_eval_metric=True
        )
        if dataset.is_multiclass:
            predict = model.predict_proba
            fit_extra_kwargs = {'eval_metric': 'merror'}
        else:
            predict = lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]  # noqa
            fit_extra_kwargs = {'eval_metric': 'error'}
    report['prediction_type'] = None if dataset.is_regression else 'probs'

    # >>> training
    logger.info('training...')
    with delu.Timer() as timer:
        model.fit(
            dataset.X_num['train'],
            dataset.Y['train'],
            eval_set=[(dataset.X_num['val'], dataset.Y['val'])],
            **C.fit,
            **fit_extra_kwargs,
        )
    report['time'] = str(timer)
    report['best_iteration'] = model.best_iteration

    # >>> finish
    model.save_model(str(output / "model.xgbm"))
    np.save(output / "feature_importances.npy", model.feature_importances_)
    predictions = {k: predict(v) for k, v in dataset.X_num.items()}
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
