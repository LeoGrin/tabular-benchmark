# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import delu
import optuna
import optuna.samplers
import optuna.trial

import lib
from lib import KWArgs


# %%
@dataclass(frozen=True)
class Config:
    seed: int
    function: Union[str, lib.Function]
    space: dict[str, Any]
    n_trials: Optional[int]
    timeout: Optional[int]
    sampler: KWArgs  # optuna.samplers.TPESampler

    def __post_init__(self):
        assert 'seed' not in self.sampler


# %%
def _suggest(trial: optuna.trial.Trial, distribution: str, label: str, *args):
    return getattr(trial, f'suggest_{distribution}')(label, *args)


def sample_config(
    trial: optuna.trial.Trial,
    space: Union[bool, int, float, str, bytes, list, dict],
    label_parts: list,
) -> Any:
    if isinstance(space, (bool, int, float, str, bytes)):
        return space
    elif isinstance(space, dict):
        if '_tune_' in space or '-tune-' in space:
            distribution = space['_tune_'] if '_tune_' in space else space['-tune-']
            # for complex cases, for example:
            # [model]
            # _tune_ = "complex-custom-distribution"
            # <any keys and values for the distribution>
            if distribution == "complex-custom-distribution":
                raise NotImplementedError()
            else:
                raise ValueError(f'Unknown distibution: "{distribution}"')
        else:
            return {
                key: sample_config(trial, subspace, label_parts + [key])
                for key, subspace in space.items()
            }
    elif isinstance(space, list):
        if not space:
            return space
        elif space[0] not in ['_tune_', '-tune-']:
            return [
                sample_config(trial, subspace, label_parts + [i])
                for i, subspace in enumerate(space)
            ]
        else:
            # space = ["_tune_"/"-tune-", distribution, distribution_arg_0, distribution_1, ...]
            _, distribution, *args = space
            label = '.'.join(map(str, label_parts))

            if distribution.startswith('?'):
                default, args_ = args[0], args[1:]
                if trial.suggest_categorical('?' + label, [False, True]):
                    return _suggest(trial, distribution.lstrip('?'), label, *args_)
                else:
                    return default

            elif distribution == '$list':
                size, item_distribution, *item_args = args
                return [
                    _suggest(trial, item_distribution, label + f'.{i}', *item_args)
                    for i in range(size)
                ]

            else:
                return _suggest(trial, distribution, label, *args)


def main(
    config: lib.JSONDict,
    output: Union[str, Path],
    *,
    force: bool = False,
    continue_: bool = False,
) -> Optional[lib.JSONDict]:
    if not lib.start(output, force=force, continue_=continue_):
        return None

    output = Path(output)
    # commented because continue_=True is not properly supported
    # logger = lib.get_logger(output / '0.log')
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    delu.random.seed(C.seed)
    function: lib.Function = (
        lib.import_(C.function) if isinstance(C.function, str) else C.function
    )

    if lib.get_checkpoint_path(output).exists():
        del report
        checkpoint = lib.load_checkpoint(output)
        report, study, trial_reports, timer = (
            checkpoint['report'],
            checkpoint['study'],
            checkpoint['trial_reports'],
            checkpoint['timer'],
        )
        delu.random.set_state(checkpoint['random_state'])
        n_trials = None if C.n_trials is None else C.n_trials - len(study.trials)
        timeout = None if C.timeout is None else C.timeout - timer()

        report.setdefault('continuations', []).append(len(study.trials))
        print(
            f'Resuming from checkpoint ({len(study.trials)} completed, {n_trials or "inf"} remaining)'
        )
        time.sleep(1.0)
    else:
        n_trials = C.n_trials
        timeout = C.timeout
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(**C.sampler, seed=C.seed),
        )
        trial_reports = []
        timer = delu.Timer()

    def objective(trial: optuna.trial.Trial) -> float:
        raw_config = sample_config(trial, C.space, [])
        with tempfile.TemporaryDirectory(suffix=f'_trial_{trial.number}') as tmp:
            report = function(raw_config, Path(tmp) / 'output')
        assert report is not None
        report['trial_id'] = trial.number
        report['tuning_time'] = str(timer)
        trial_reports.append(report)
        delu.cuda.free_memory()
        return report['metrics']['val']['score']

    def callback(*_, **__):
        report['best'] = trial_reports[study.best_trial.number]
        report['time'] = str(timer)
        report['n_completed_trials'] = len(trial_reports)
        lib.dump_checkpoint(
            {
                'report': report,
                'study': study,
                'trial_reports': trial_reports,
                'timer': timer,
                'random_state': delu.random.get_state(),
            },
            output,
        )
        lib.dump_summary(lib.summarize(report), output)
        lib.dump_report(report, output)
        lib.backup_output(output)
        print(f'Time: {timer}')

    timer.run()
    # ignore the progress bar warning
    warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
    # ignore the warnings about the deprecated suggest_* methods
    warnings.filterwarnings('ignore', category=FutureWarning)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[callback],
        show_progress_bar=True,
    )
    lib.dump_summary(lib.summarize(report), output)
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)
