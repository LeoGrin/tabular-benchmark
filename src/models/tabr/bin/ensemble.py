# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import shutil
from dataclasses import replace
from pathlib import Path

import delu
import numpy as np
from loguru import logger
from scipy.special import expit, softmax

import lib


def main(
    evaluation_dir: Path,
    n_ensembles: int = 3,
    ensemble_size: int = 5,
    *,
    force: bool = False,
):
    evaluation_dir = lib.get_path(evaluation_dir)
    assert evaluation_dir.name.endswith('-evaluation')

    for ensemble_id in range(n_ensembles):
        seeds = range(ensemble_id * ensemble_size, (ensemble_id + 1) * ensemble_size)
        single_outputs = [evaluation_dir / str(x) for x in seeds]
        output = evaluation_dir.with_name(
            evaluation_dir.name.replace('evaluation', f'ensemble-{ensemble_size}')
        ) / str(ensemble_id)
        relative_output = lib.try_get_relative_path(output)
        if not all((x / 'DONE').exists() for x in single_outputs):
            logger.warning(f'Not enough single models | {relative_output}')
            continue
        if output.exists():
            if force:
                logger.warning(f'Removing the existing output | {relative_output}')
                shutil.rmtree(output)
            else:
                lib.backup_output(output)
                logger.warning(f"Already exists! | {relative_output}")
                continue
        del relative_output

        output.mkdir(parents=True)
        report: lib.JSONDict = {
            'function': 'bin.ensemble.main',
            'config': {
                'seeds': list(seeds),
            },
        }
        delu.random.seed(0)

        first_report = lib.load_report(single_outputs[0])
        report['single_model_function'] = first_report['function']
        report['data'] = first_report["config"]["data"]["path"]
        dataset = lib.Dataset.from_dir(
            lib.get_path(first_report['config']['data']['path']),
            first_report['config']['data'].get('score'),
        )
        report['prediction_type'] = None if dataset.is_regression else 'probs'
        y_policy = first_report['config']['data']['y_policy']
        if y_policy is not None:
            dataset.data['Y'], y_info = lib.transform_y(
                dataset.Y, lib.YPolicy(first_report['config']['data']['y_policy'])
            )
            dataset = replace(dataset, y_info=y_info)
        single_predictions = [lib.load_predictions(x) for x in single_outputs]

        predictions = {}
        for part in ['train', 'val', 'test']:
            stacked_predictions = np.stack([x[part] for x in single_predictions])  # type: ignore[code]
            if dataset.is_binclass:
                if first_report['prediction_type'] == 'logits':
                    stacked_predictions = expit(stacked_predictions)
            elif dataset.is_multiclass:
                if first_report['prediction_type'] == 'logits':
                    stacked_predictions = softmax(stacked_predictions, -1)
            else:
                assert dataset.is_regression
            predictions[part] = stacked_predictions.mean(0)
        report['metrics'] = dataset.calculate_metrics(
            predictions, report['prediction_type']
        )
        lib.dump_predictions(predictions, output)
        lib.dump_summary(lib.summarize(report), output)
        lib.finish(output, report)


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_cli(main)
