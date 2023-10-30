# LightGBM

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
from lightgbm import LGBMClassifier, LGBMRegressor
from loguru import logger

import lib
from lib import KWArgs


@dataclass(frozen=True)
class Config:
    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]  # lib.data.build_dataset
    model: KWArgs  # lightgbm.{LGBMClassifier,LGBMRegressor}
    fit: KWArgs  # lightgbm.{LGBMClassifier,LGBMRegressor}.fit

    def __post_init__(self):
        assert 'random_state' not in self.model
        if isinstance(self.data, dict):
            assert self.data['cat_policy'] in [None, 'one-hot']
        assert 'early_stopping_rounds' in self.fit
        assert 'device' not in self.model, 'Use "device_type" argument instead.'
        if self.model.get('device_type') == 'gpu':
            # In fact, in our environment, LightGBM does not support GPU.
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
    assert set(dataset.data.keys()) == {'X_num', 'Y'}, set(dataset.data.keys())
    assert dataset.X_num is not None  # for type checker

    # >>> model
    model_extra_kwargs = {'random_state': C.seed}
    if dataset.is_regression:
        model = LGBMRegressor(**C.model, **model_extra_kwargs)
        fit_extra_kwargs = {'eval_metric': 'rmse'}
        predict = model.predict
    else:
        model = LGBMClassifier(**C.model, **model_extra_kwargs)
        if dataset.is_multiclass:
            predict = model.predict_proba
            fit_extra_kwargs = {'eval_metric': 'multi_error'}
        else:
            predict = lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]  # noqa
            fit_extra_kwargs = {'eval_metric': 'binary_error'}
    report['prediction_type'] = None if dataset.is_regression else 'probs'

    # >>> training
    logger.info('training...')
    with delu.Timer() as timer:
        model.fit(
            dataset.X_num['train'],
            dataset.Y['train'],
            eval_set=(dataset.X_num['val'], dataset.Y['val']),
            **C.fit,
            **fit_extra_kwargs,
        )
    report['time'] = str(timer)
    report['best_iteration'] = model.booster_.best_iteration

    # >>> finish
    lib.dump_pickle(model, output / 'model.pickle')
    np.save(output / 'feature_importances.npy', model.feature_importances_)
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
